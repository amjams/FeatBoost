"""
FeatBoost for Classification
"""

# Author: Ahmad Alsahaf <a.m.j.a.alsahaf@rug.nl>
# Vikram Shenoy <shenoy.vi@husky.neu.edu>

import itertools
import math
import sys
import warnings
from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold


class FeatBoostClassifier(BaseEstimator):
    """
        FeatBoost For Classification Problems

    Parameters
    ----------
    estimator : object OR list of objects, List shape = 2 ([e1, e2])
        A tree based estimator, with a 'fit' method that returns the
                attribute 'feature_importances_'.
                (In the case of XGBoost, 'get_scores()').
                - If only one estimator is specified, the same estimator is used for
                the Ranking step, the SISO step, and the MISO step.
                - If a list of estimators is provided i.e. [e1, e2], then e1 is
                used for the Ranking step, e2 is used for the SISO step, e1 is used
                for the MISO step.
                Estimators supported:
                1) XGBoost Classifier
                2) Random Forest Classifier
                3) Extra Trees Classifier

        number_of_folds : int, Optional (default=5)
                The number of folds for K-fold cross validation.

        epsilon : int, Optional (default=1e-18)
                The threshold value which determines one of the stopping conditions
                for FeatBoost.
                Ideally this value needs to be extremely small. It could be negative.

        max_number_of_features : int, Optional (default = 10)
                The maximum number of features to be selected.
                The algorithm returns a feature subset of size less than or equal to "max_number_of_features" amount of

        siso_ranking_size : int OR list  Optional (default=5)
                The number of variables evaluated at each step.
                The first 'siso_ranking_size' variables after Input Ranking are used
                to determine the selected variable. Corresponds to parameters /
                'm' in the paper.

                -> int type:
                If it takes one value (e.g. 5, the current default value), it
                operates normally, evaluating SISO for the top 5 ranked features.

                -> list type:
                If it takes two values as a list (e.g. [5 10]), it evaluates SISO
                for 5 random variables selected from the top 10.

        siso_order : int  Optional (default=1)
                Corresponds to the size of feature combinations evaluated at the
                SISO step. We recommend keeping this to 1.

        global_sample_weights : array, shape = [Y], Optional (default=None)
                The initial weights of the sample set.

        loss: String, Optional (default="softmax")
                Specifies the loss function to be used, which in turn affects how
                sample weights are updates
                Options:
                1) Softmax Loss(Categorical Cross-Entropy loss) = "softmax"
                2) Adaptive Boosting = "adaboost"

        reset: Boolean, Optional (default=True)
                If set to True, the reset option allows the assignment of initial values
                to sample weights when the boosting process fails to find new useful
                features. See the paper for more details.

        fast_mode: Boolean, Optional (default=False)
                If false, for every SISO iteration, we append
                the new top ranked features to the features selected so far
                (with default weights) for evaluation.

                If True, the new top ranked variables are evaluated alone with
                each model being tested on a weighted sample distribution.
                Could lead to premature stopping.

        metric: String, Optional (default='acc')
                The evaluation metric for selecting the best feature. The default metric
                is classification accuracy. 'f1' is the other available option.

        xgb_importance: String, Optional (default='gain')
                The XGBoost Importance Type field. Importance type can be defined as:
                'weight': the number of times a feature is used to split the data across
                                  all trees.
                'gain': the average gain across all splits the feature is used in.
                'cover': the average coverage across all splits the feature is used in.
                'total_gain': the total gain across all splits the feature is used in.
                'total_cover': the total coverage across all splits the feature is used
                                           in.
                For more details, read the XGBoost documentation linked below.
                <https://xgboost.readthedocs.io/en/latest/python/python_api.html>

        learning_rate: float, Optional (default=1.0)
                The rate at which the weights are updated.

    verbose : int, Optional (default=0)
        Controls verbosity of output:
        - 0: No Output
        - 1: Displays selected features in each iteration
        - 2: Displays Folds along with selected features

        Attributes
    ----------
        selected_subset_ : array, max size = max_number_of_features
                Returns an array of selected features.

        accuracy_ : array, shape = [selected_subset_].
                Returns an array of the accuracy of the corresponding selected
                features.

        stopping_condition_ : String
                Specifies what caused the selection of features to stop.
                1) max_number_of_features_reached -> The size of the subset is equal
                                                                                        to max_number_of_features
                2) tolerance_reached -> If the difference between the accuracy goes
                                                                below the threshold, the selection of
                                                                features stops.
                3) variable_selected_twice -> If a variable has already been
                                                                          selected, feature selection stops.

        residual_weights_ : array, shape = [max_number_of_features, Y]
                Returns the updated weights of all the samples after performing
                FeatBoost.

        siso_ranking_ : array, shape = [max_number_of_features, siso_ranking_size]
                Returns the top siso_ranking_size number of features in each internal
                iteration. Each row correpsonds to the number of internal iterations
                and each column corresponds to the rank of the feature for that
                iteration.

        feature_importances_array_: BETA
                Returns the feature importance scores after each iteration
    """

    def __init__(
        self,
        estimator,
        number_of_folds=10,
        epsilon=1e-18,
        max_number_of_features=10,
        siso_ranking_size=5,
        siso_order=1,
        loss="softmax",
        reset=True,
        fast_mode=False,
        metric="acc",
        xgb_importance="gain",
        learning_rate=1,
        verbose=0,
    ):
        if type(estimator) is list:
            assert (
                len(estimator) == 2
            ), "Length of list of estimators should always be equal to 2.\nRead the documentation for more details"
            self.estimator = estimator
        else:
            self.estimator = [estimator, estimator]
        self.number_of_folds = number_of_folds
        self.epsilon = epsilon
        self.max_number_of_features = max_number_of_features
        self.siso_ranking_size = siso_ranking_size
        self.siso_order = siso_order
        if type(self.siso_ranking_size) is list:
            assert (
                self.siso_ranking_size[0] > self.siso_order
            ), "SISO order cannot be greater than the SISO ranking size.\nRead the documentation for more details"
        else:
            assert (
                self.siso_ranking_size > self.siso_order
            ), "SISO order cannot be greater than the SISO ranking size.\nRead the documentation for more details"
        self.loss = loss
        self.reset = reset
        self.fast_mode = fast_mode
        self.metric = metric
        self.xgb_importance = xgb_importance
        self.learning_rate = learning_rate
        self.verbose = verbose

    def fit(self, X, Y):
        """
        Fits the FeatBoost method with the estimator as provided by the user.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        Y : array-like, shape = [n_samples]
            The target values.

                Returns
        -------
        self : object

        """
        return self._fit(X, Y)

    def transform(self, X):
        """
                Reduces the columns of input X to the features selected by FeatBoost.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features selected by
                        FeatBoost.
        """
        return self._transform(X)

    def fit_transform(self, X, Y):
        """
                Fits FeatBoost and then reduces the input X to the features selected by FeatBoost.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        Y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features selected by
                        FeatBoost.
        """
        return self._fit_transform(X, Y)

    def _fit(self, X, Y, feature_names=None, global_sample_weights=None):
        """
        Performs the Initial ranking, SISO and MISO over multiple iterations
        based on the maximum number of features required by the user in a single
        subset.
        """
        self._n_classes_ = len(np.unique(Y))
        self._feature_names = feature_names
        self.feature_importances_array_ = np.empty((0, X.shape[1]))
        # Give features a default name.
        if self._feature_names is None:
            self._feature_names = []
            for i in range(len(X[0])):
                temp_name = "x_%03d" % (i + 1)
                self._feature_names.append(temp_name)
        self._global_sample_weights = global_sample_weights
        # Initialize sample weights.
        if self._global_sample_weights is None:
            self._global_sample_weights = np.ones(np.shape(Y))
            self.residual_weights_ = np.zeros((self.max_number_of_features, len(Y)))
            if type(self.siso_ranking_size) is int:
                self.siso_ranking_ = 99 * np.ones(
                    (self.max_number_of_features, self.siso_ranking_size)
                )
            elif type(self.siso_ranking_size) is list:
                assert (
                    len(self.siso_ranking_size) == 2
                ), "siso_ranking_size of list type is of incompatible format. Please enter a list of the following type: \n siso_ranking_size=[5, 10] \n Read documentation for more details."
                self.siso_ranking_ = 99 * np.ones(
                    (self.max_number_of_features, self.siso_ranking_size[0])
                )

        stop_epsilon = 10e6
        iteration_number = 1
        self._all_selected_variables = []
        self.accuracy_ = []
        repeated_variable = False

        # alpha intialization (for normalizing later)
        if self.loss == "adaboost":
            self._alpha = np.ones(self.max_number_of_features + 1)
            self._alpha_abs = np.ones(self.max_number_of_features + 1)
        elif self.loss == "softmax":
            self._alpha = np.ones((len(Y), self.max_number_of_features + 1))
            self._alpha_abs = np.ones((len(Y), self.max_number_of_features + 1))

        # loop counter for reset
        reset_count = 0
        while (
            stop_epsilon > self.epsilon
            and iteration_number <= self.max_number_of_features
            and repeated_variable is False
        ):
            if iteration_number == 1:
                if self.verbose > 0:
                    print(
                        "\n\n\n\n\n\nRanking features iteration %02d"
                        % (iteration_number)
                    )
                # Perform Single Input Single Output (SISO) for iteration 1.
                selected_variable, best_acc_t = self._siso(X, Y, iteration_number)
                if self.verbose > 1:
                    print("Evaluating MISO after iteration %02d" % (iteration_number))
                # The selected feature is stored inside self._all_selected_variables.
                self._all_selected_variables.extend(selected_variable)
                # Perform Multiple Input Single Output (MISO) for iteration 1.
                acc_t_miso = self._miso(
                    X[:, self._all_selected_variables], Y, iteration_number
                )
                # Accuracy of selected feature is stored in accuracy_.
                self.accuracy_.append(acc_t_miso)
                if self.verbose > 1:
                    print(
                        "::::::::::::::::::::accuracy of MISO after iteration %02d is %05f"
                        % (iteration_number, acc_t_miso)
                    )
                iteration_number = iteration_number + 1
            else:
                if reset_count >= 1:
                    self.reset = False

                    print("Infinite loop: No more resets this time!")
                    reset_count = 0  # reset the reset counter!
                temp_str = [
                    self._feature_names[i] for i in self._all_selected_variables
                ]
                if self.verbose > 0:
                    print(
                        "\n\n\n\n\nselected variable thus far:\n%s"
                        % "\n".join(temp_str)
                    )
                    print("Ranking features iteration %02d" % (iteration_number))
                # Perform Single Input Single Output (SISO) for subsequent iterations.
                selected_variable, best_acc_t = self._siso(X, Y, iteration_number)
                if self.verbose > 1:
                    print("Evaluating MISO after iteration %02d" % (iteration_number))
                # Check if the feature has already been selected i.e. stopping condition 3 as mentioned above.
                # if selected_variable in self._all_selected_variables:
                if all(x in self._all_selected_variables for x in selected_variable):
                    repeated_variable = True
                else:
                    # The selected feature is stored inside self._all_selected_variables.
                    for x in selected_variable:
                        if x not in self._all_selected_variables:
                            self._all_selected_variables.extend([x])

                    # Perform Multiple Input Single Output (MISO) for subsequent iterations.
                    acc_t_miso = self._miso(
                        X[:, self._all_selected_variables], Y, iteration_number
                    )
                    # Accuracy of selected features is stored in accuracy_.
                    self.accuracy_.append(acc_t_miso)
                    # stop_epsilon makes sure the accuracy doesn't fall below the threshold i.e stopping condition 2 as mentioned above.
                    stop_epsilon = (
                        self.accuracy_[iteration_number - 1]
                        - self.accuracy_[iteration_number - 2]
                    )
                    if self.verbose > 1:
                        print(
                            "::::::::::::::::::::accuracy of MISO after iteration %02d is %05f"
                            % (iteration_number, acc_t_miso)
                        )
                    if reset_count > 0 and stop_epsilon > self.epsilon:
                        reset_count = 0  # reset the reset counter!
                        print(
                            "CONGRATURLATIONS: A reset was successful! The reset counter has been reset."
                        )

                    iteration_number = iteration_number + 1
            # Stopping Condtion 1 -> Maximum number of features reached.
            if iteration_number > self.max_number_of_features:
                if self.verbose > 0:
                    print(
                        "Selection stopped: Maximum number of iteration %02d has been reached."
                        % (self.max_number_of_features)
                    )
                self.stopping_condition_ = "max_number_of_features_reached"
                self.selected_subset_ = self._all_selected_variables

            # Stopping Condtion 2 -> epsilon value falls below the threshold.
            if stop_epsilon <= self.epsilon:
                if self.reset is False:
                    if self.verbose > 0:
                        print("Selection stopped: Tolerance has been reached.")
                    print(
                        "Stopping Condition triggered at iteration number: %d"
                        % (iteration_number - 1)
                    )
                    self.stopping_condition_ = "tolerance_reached"
                    self.selected_subset_ = self._all_selected_variables[:-1]
                    self.complete_subset_ = self._all_selected_variables[:-1]
                    self.accuracy_ = self.accuracy_[:-1]

                    print("Selected variables so far:")
                    print(self.selected_subset_)

                # Reset condition triggered
                elif self.reset is True:
                    # re-set the sample weights and epsilon
                    print("\n\nATTENTION: Reset occured because of tolerance reached!")
                    stop_epsilon = self.epsilon + 1
                    if reset_count == 0:
                        self._global_sample_weights = np.ones(np.shape(Y))
                    elif reset_count == 1:  # We don't do this anymore
                        self._global_sample_weights = np.random.randn(len(Y))
                        self._global_sample_weights = (
                            self._global_sample_weights
                            / np.sum(self._global_sample_weights)
                            * len(Y)
                        )
                    reset_count += 1

                    # undoing the iteration
                    iteration_number = iteration_number - 1
                    self._all_selected_variables = self._all_selected_variables[:-1]
                    self.accuracy_ = self.accuracy_[:-1]

            # Stopping Condtion 3 -> A specific feature has been already selected previously.
            if repeated_variable:
                if self.reset is False:
                    if self.verbose > 0:
                        print("Selection stopped: A variable has been selected twice.")
                    print(
                        "Stopping Condition triggered at iteration number: %d"
                        % (iteration_number - 1)
                    )
                    self.stopping_condition_ = "variable_selected_twice"
                    self.selected_subset_ = self._all_selected_variables[:]
                    self.complete_subset_ = self._all_selected_variables[:]
                    self.accuracy_ = self.accuracy_

                elif self.reset is True:
                    # re-set the sample weights and epsilon
                    print("\n\nATTENTION: Reset occured because of selected twice!")
                    repeated_variable = False
                    if reset_count == 0:
                        self._global_sample_weights = np.ones(np.shape(Y))
                    elif reset_count == 1:
                        self._global_sample_weights = np.random.randn(len(Y))
                        self._global_sample_weights = (
                            self._global_sample_weights
                            / np.sum(self._global_sample_weights)
                            * len(Y)
                        )
                    reset_count += 1

                    # undoing the iteration
                    iteration_number = iteration_number - 1
                    self._all_selected_variables = self._all_selected_variables[:]
                    self.accuracy_ = self.accuracy_[:]

    def _siso(self, X, Y, iteration_number):
        """
        Determines which feature to select based on classification accuracy of
        the 'siso_ranking_size' ranked features from _input_ranking.
        """
        warnings.filterwarnings("ignore")
        # Get a ranking of features based on the estimator.
        ranking, self.all_ranking_ = self._input_ranking(X, Y, iteration_number)
        self.siso_ranking_[(iteration_number - 1), :] = ranking
        kf = KFold(n_splits=self.number_of_folds, shuffle=True, random_state=275)
        # Combination of features from the ranking up to siso_order size
        combs = []
        for i in range(self.siso_order):
            temp_comb = [list(x) for x in itertools.combinations(ranking, i + 1)]
            combs.extend(temp_comb)

        acc_t_all = np.zeros((len(combs), 1))
        std_t_all = np.zeros((len(combs), 1))
        for idx_1, i in enumerate(combs):
            if self.verbose > 1:
                print(
                    "...Evaluating SISO combination %02d which is %s"
                    % (idx_1 + 1, str(i))
                )
            X_temp = X[:, i]
            n = len(X_temp)
            X_temp = X_temp.reshape(n, len(i))

            if self.fast_mode is False:
                X_temp = np.concatenate(
                    (X_temp, X[:, self._all_selected_variables]), axis=1
                )

            count = 1
            acc_t_folds = np.zeros((self.number_of_folds, 1))
            # Compute accuracy for each SISO input.
            for train_index, test_index in kf.split(X_temp):
                X_train, X_test = X_temp[train_index], X_temp[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                # fit model according to mode
                if self.fast_mode is False:
                    self.estimator[1].fit(X_train, np.ravel(y_train))
                else:
                    self.estimator[1].fit(
                        X_train,
                        np.ravel(y_train),
                        sample_weight=self._global_sample_weights[train_index],
                    )

                yHat_test = self.estimator[1].predict(X_test)
                # calculate the required metric
                if self.metric == "acc":
                    acc_t = accuracy_score(y_test, yHat_test)
                elif self.metric == "f1":
                    acc_t = f1_score(y_test, yHat_test, average="weighted")
                if self.verbose > 1:
                    print("Fold %02d accuracy = %05f" % (count, acc_t))
                acc_t_folds[count - 1, :] = acc_t
                count = count + 1
            acc_t_all[idx_1, :] = np.mean(acc_t_folds)
            std_t_all[idx_1, :] = np.std(acc_t_folds)
            if self.verbose > 1:
                print(
                    "accuracy for combination %02d is = %05f"
                    % (idx_1 + 1, np.mean(acc_t_folds))
                )

        # regular
        best_acc_t = np.amax(acc_t_all)
        selected_variable = combs[np.argmax(acc_t_all)]

        if self.verbose > 1:
            print(
                "Selected variable is %s with accuracy %05f"
                % (str(selected_variable), best_acc_t)
            )
        return selected_variable, best_acc_t

    def _miso(self, X, Y, iteration_number):
        """
        Calculates the accuracy of selected features one additional feature at a
        time and also computes the updated weights of the samples.
        """
        warnings.filterwarnings("ignore")
        kf = KFold(n_splits=self.number_of_folds, shuffle=True, random_state=275)
        count = 1
        acc_t_folds = np.zeros((self.number_of_folds, 1))
        # Compute the accuracy of the selected features one addition at a time.
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            self.estimator[1].fit(X_train, np.ravel(y_train))  # changed estimator!
            yHat_test = self.estimator[1].predict(X_test)
            # find performance based on selected metric
            if self.metric == "acc":
                acc_t = accuracy_score(y_test, yHat_test)
            elif self.metric == "f1":
                acc_t = f1_score(y_test, yHat_test, average="weighted")

            if self.verbose > 1:
                print("Fold %02d accuracy = %05f" % (count, acc_t))
            acc_t_folds[count - 1, :] = acc_t
            count = count + 1
        acc_t_miso = np.mean(acc_t_folds)
        # Calculate the residual weights from fitting on the entire dataset.
        self.estimator[0].fit(X, np.ravel(Y))
        yHat_train_full = self.estimator[0].predict(X)
        if self.loss == "adaboost":
            # Determine the missclassified samples.
            acc_train_full = accuracy_score(Y, yHat_train_full)
            err = 1 - acc_train_full
            self._alpha_abs[iteration_number] = np.log((1 - err) / err) + np.log(
                self._n_classes_ - 1
            )
            self._alpha[iteration_number] = np.divide(
                self._alpha_abs[iteration_number], self._alpha_abs[iteration_number - 1]
            )
            misclass = np.subtract(
                Y.reshape(len(Y), 1), yHat_train_full.reshape(len(Y), 1)
            )
            misclass_idx = np.nonzero(misclass)
            misclass_idx = misclass_idx[0]
            # Get the correct classification.
            corclass_idx = np.nonzero(misclass == 0)
            corclass_idx = corclass_idx[0]
            # Weighting up/down misclassified/classified samples.
            self._global_sample_weights[misclass_idx] = self._global_sample_weights[
                misclass_idx
            ] * np.exp(self._alpha[iteration_number])

            # re-normalize
            self._global_sample_weights = (
                self._global_sample_weights
                / np.sum(self._global_sample_weights)
                * len(Y)
            )
            print("mean of weights = %02f" % np.mean(self._global_sample_weights))
            print(self._global_sample_weights[0:10])
            if iteration_number == 1:
                self.residual_weights_[(iteration_number - 1), misclass_idx] = (
                    self.residual_weights_[(iteration_number - 1), misclass_idx] + 1
                )
            else:
                self.residual_weights_[
                    (iteration_number - 1), :
                ] = self.residual_weights_[(iteration_number - 2), :]
                self.residual_weights_[(iteration_number - 1), misclass_idx] = (
                    self.residual_weights_[(iteration_number - 1), misclass_idx] + 1
                )

        elif self.loss == "softmax":
            # Determine the missclassified samples.
            if iteration_number == 1:
                self.residual_weights_[(iteration_number - 1), :] = 0
            else:
                self.residual_weights_[
                    (iteration_number - 1), :
                ] = self._global_sample_weights
            # Gets all the labels.
            labels = np.unique(np.ravel(Y))
            Y_class = np.zeros((len(Y), len(labels)))
            prediction_probabiltiy = self.estimator[0].predict_proba(X)
            probability_weight = np.zeros(np.shape(Y))
            # Generates One-Hot encodings for Multi-Class Problems
            for i in range(0, len(X)):
                for j in range(0, len(labels)):
                    if Y[i] == labels[j]:
                        if len(labels) == 2:
                            Y[i] = j
                            probability_weight[i] = prediction_probabiltiy[i][j]
                        Y_class[i][j] = 1
            log_bias = 1e-30

            # Loss function
            self._alpha_abs[:, iteration_number] = -self.learning_rate * np.sum(
                Y_class * np.log(prediction_probabiltiy + log_bias), axis=1
            )
            self._alpha[:, iteration_number] = np.divide(
                self._alpha_abs[:, iteration_number],
                self._alpha_abs[:, iteration_number - 1],
            )
            self._global_sample_weights = (
                self._global_sample_weights * self._alpha[:, iteration_number]
            )
            # re-normalize
            self._global_sample_weights = (
                self._global_sample_weights
                / np.sum(self._global_sample_weights)
                * len(Y)
            )
            # self._global_sample_weights = np.ones(np.shape(Y))*self._alpha[:,iteration_number]
            self.residual_weights_[
                (iteration_number - 1), :
            ] = self._global_sample_weights
        return acc_t_miso

    def _input_ranking(self, X, Y, iteration_number):
        """
        Creates an initial ranking of features using the provided estimator for
        SISO evaluation.
        """
        # Perform an initial ranking of features using the given estimator.
        check_estimator = str(self.estimator[0])
        if "XGBClassifier" in check_estimator:
            self.estimator[0].fit(
                X, np.ravel(Y), sample_weight=self._global_sample_weights
            )
            fscore = (
                self.estimator[0]
                .get_booster()
                .get_score(importance_type=self.xgb_importance)
            )
            feature_importance = np.zeros(X.shape[1])
            self.feature_importances_array_ = np.vstack(
                (self.feature_importances_array_, feature_importance)
            )
            for k, v in fscore.items():
                feature_importance[int(k[1:])] = v
            feature_rank = np.argsort(feature_importance)
            all_ranking = feature_rank[::-1]
            if self.verbose > 1:
                print("feature importances of all available feature:")
            count = 0
            if type(self.siso_ranking_size) is int:
                for i in range(-1, -1 * self.siso_ranking_size - 1, -1):
                    if self.verbose > 1:
                        print(
                            "%s   %05f"
                            % (
                                self._feature_names[feature_rank[i]],
                                feature_importance[feature_rank[i]],
                            )
                        )
                    count = count + 1
                # Return the 'siso_ranking_size' ranked features to perform SISO.
                return (
                    feature_rank[: -1 * self.siso_ranking_size - 1 : -1],
                    all_ranking,
                )

            elif type(self.siso_ranking_size) is list:
                assert (
                    len(self.siso_ranking_size) == 2
                ), "siso_ranking_size of list type is of incompatible format. Please enter a list of the following type: \n siso_ranking_size=[5, 10] \n Read documentation for more details."
                for i in range(-1, -1 * self.siso_ranking_size[1] - 1, -1):
                    if self.verbose > 1:
                        print(
                            "%s   %05f"
                            % (
                                self._feature_names[feature_rank[i]],
                                feature_importance[feature_rank[i]],
                            )
                        )
                    count = count + 1
                # Return the 'siso_ranking_size' ranked features to perform SISO.
                feature_rank = feature_rank[: -1 * self.siso_ranking_size[1] - 1 : -1]
                return (
                    np.random.choice(
                        feature_rank, self.siso_ranking_size[0], replace=False
                    ),
                    all_ranking,
                )
        else:
            self.estimator[0].fit(
                np.nan_to_num(X),
                np.nan_to_num(np.ravel(Y)),
                sample_weight=np.nan_to_num(self._global_sample_weights),
            )
            feature_importance = self.estimator[0].feature_importances_
            self.feature_importances_array_ = np.vstack(
                (self.feature_importances_array_, feature_importance)
            )
            feature_rank = np.argsort(feature_importance)
            all_ranking = feature_rank[::-1]
            if self.verbose > 1:
                print("feature importances of all available feature:")
            if type(self.siso_ranking_size) is int:
                for i in range(-1, -1 * self.siso_ranking_size - 1, -1):
                    if self.verbose > 1:
                        print(
                            "%s   %05f"
                            % (
                                self._feature_names[feature_rank[i]],
                                feature_importance[feature_rank[i]],
                            )
                        )
                return (
                    feature_rank[: -1 * self.siso_ranking_size - 1 : -1],
                    all_ranking,
                )
            elif type(self.siso_ranking_size) is list:
                assert len(self.siso_ranking_size) == 2, "/* SISO CONDITION */"
                for i in range(-1, -1 * self.siso_ranking_size[1] - 1, -1):
                    if self.verbose > 1:
                        print(
                            "%s   %05f"
                            % (
                                self._feature_names[feature_rank[i]],
                                feature_importance[feature_rank[i]],
                            )
                        )
                feature_rank = feature_rank[: -1 * self.siso_ranking_size[1] - 1 : -1]
                return (
                    np.random.choice(
                        feature_rank, self.siso_ranking_size[0], replace=False
                    ),
                    all_ranking,
                )

    def _transform(self, X):
        # Check if the fit(X, Y) function has been called prior to performing transform(X)
        try:
            self.selected_subset_
        except AttributeError:
            raise ValueError("fit(X, Y) needs to be called before using transform(X).")
        return X[:, self.selected_subset_]

    def _fit_transform(self, X, Y):
        self._fit(X, Y)
        return self._transform(X)
