import math
import sys
import itertools
import warnings
from warnings import warn

import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class IISClassification():
	"""
		Iterative Input Selection For Classification Problems

	    Parameters
	    ----------
	    estimator : object OR list of objects, List shape = 3 ([e1, e2, e3])
	        A tree based estimator, with a 'fit' method that returns the
			attribute 'feature_importances_'.
			(In the case of XGBoost, 'get_scores()').
			- If only one estimator is specified, the same estimator is used for
			the Ranking step, the SISO step, and the MISO step.
			- If a list of estimators is provided i.e. [e1, e2, e3], then e1 is
			used for the Ranking step, e2 is used for the SISO step, e3 is used
			for the MISO step.
			Estimators supported:
			1) XGBoost Classifier
			2) Random Forest Classifier
			3) Extra Trees Classifier

		number_of_folds : int, Optional (default  = 5)
			The number of folds for K-fold cross validation.

		epsilon : int, Optional (default = 1e-18)
			The threshold value which determines one of the stopping conditions
			for Iterative Input Selection for Classification.
			Ideally this value needs to be extremely small. It could be negative.

		max_number_of_features : int, Optional (default = 10)
			Corresponds to the maximum number of features allowed in a subset.
			The algorithm does not return "max_number_of_features" amount of
			features, but returns upto number of features upto this value.

		siso_ranking_size : int OR list  Optional (default=5 or default=[5, 10])
			Corresponds to the number of variables considered for the SISO step.
			The first 'siso_ranking_size' variables after Input Ranking are used
			to determine the selected variable.

			-> int type:
			If it takes one value (e.g. 5, the current default value), it
			operates normally, evaluating SISO for the top 5 ranked features.

			-> list type:
			If it takes two values as a list (e.g. [5 10]), it evaluates SISO
			for 5 random variables selected from the top 10.

		global_sample_weights : array, shape = [Y], Optional (default = None)
			The initial weights of the sample set. The weights are updated in
			each internal iteration using concepts of AdaBoosting.

		loss: String, Optional (default = "softmax")
			Specifies the loss function to be used which in turn affects the
			updation of weights.
			Options:
			1) Softmax Loss(Categorical Cross-Entropy loss) = "softmax"
			2) Binary Cross-Entropy loss = "binary_crossentropy"
			3) Adaptive Boosting = "adaboost"

			Note: The Binary Cross-Entropy Loss only works for Binary class
			problems.

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
			IIS.

		siso_ranking_ : array, shape = [max_number_of_features, siso_ranking_size]
			Returns the top siso_ranking_size number of features in each internal
			iteration. Each row correpsonds to the number of internal iterations
			and each column corresponds to the rank of the feature for that
			iteration.
	"""

	def __init__(self, estimator, number_of_folds=10, epsilon=1e-18, max_number_of_features=10, siso_ranking_size=5,siso_order=1, global_sample_weights=None, loss="softmax", verbose=0):
		if type(estimator) is list:
			assert len(estimator) == 3, ("Length of list of estimators should always be equal to 3.\nRead the documentation for more details")
			self.estimator = estimator
		else:
			self.estimator = [estimator, estimator, estimator]
		self.number_of_folds = number_of_folds
		self.epsilon = epsilon
		self.max_number_of_features = max_number_of_features
		self.siso_ranking_size = siso_ranking_size
		self.siso_order = siso_order
		self.loss = loss
		self.verbose = verbose

	def fit(self, X, Y):
		"""
        Fits the IISC method with the estimator as provided by the user.

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
		Reduces the columns of input X to the features selected by IISC.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features selected by
			IISC.
		"""
		return self._transform(X)

	def fit_transform(self, X, Y):
		"""
		Fits IISC and then reduces the input X to the features selected by IISC.

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
			IISC.
		"""
		return self._fit_transform(X, Y)

	def _fit(self, X, Y, feature_names=None, global_sample_weights=None):
		"""
		Performs the Initial ranking, SISO and MISO over multiple iterations
		based on the maximum number of features required by the user in a single
		subset.
		"""
		self.n_classes_ = len(np.unique(Y))
		if (self.loss == "binary_crossentropy"):
			assert self.n_classes_ == 2, ("Binary Cross-Entropy only works for Binary Class problems.\nRead the documentation for more details.")
		self.feature_names = feature_names
		# Give features a default name.
		if self.feature_names is None:
			self.feature_names = []
			for i in range(len(X[0])):
				temp_name = "x_%03d" % (i+1)
				self.feature_names.append(temp_name)
		self.global_sample_weights = global_sample_weights
		# Initialize sample weights.
		if self.global_sample_weights is None:
			self.global_sample_weights = np.ones(np.shape(Y))
			self.residual_weights_ = np.zeros((self.max_number_of_features,len(Y)))
			if type(self.siso_ranking_size) is int:
				self.siso_ranking_ = 99*np.ones((self.max_number_of_features,self.siso_ranking_size))
			elif type(self.siso_ranking_size) is list:
				assert len(self.siso_ranking_size) == 2, ("siso_ranking_size of list type is of incompatible format. Please enter a list of the following type: \n siso_ranking_size=[5, 10] \n Read documentation for more details.")
				self.siso_ranking_ = 99*np.ones((self.max_number_of_features,self.siso_ranking_size[0]))

		stop_epsilon = 10e6
		iteration_number = 1
		self.all_selected_variables = []
		self.accuracy_ = []
		repeated_variable = False
		while stop_epsilon > self.epsilon and iteration_number <= self.max_number_of_features and repeated_variable is False:
			if iteration_number == 1:
				if(self.verbose > 0):
					print "\n\n\n\n\n\nRanking features iteration %02d" % (iteration_number)
				# Perform Single Input Single Output (SISO) for iteration 1.
				selected_variable,best_acc_t = self._siso(X,Y,iteration_number)
				if(self.verbose > 1):
					print "Evaluating MISO after iteration %02d" % (iteration_number)
				# The selected feature is stored inside self.all_selected_variables.
				self.all_selected_variables.extend(selected_variable)
				# Perform Multiple Input Single Output (MISO) for iteration 1.
				acc_t_miso = self._miso(X[:, self.all_selected_variables], Y, iteration_number)
				# Accuracy of selected feature is stored in accuracy_.
				self.accuracy_.append(acc_t_miso)
				if(self.verbose > 1):
					print "::::::::::::::::::::accuracy of MISO after iteration %02d is %05f" % (iteration_number,acc_t_miso)
				iteration_number = iteration_number + 1
			else:
				temp_str = [self.feature_names[i] for i in self.all_selected_variables]
				if(self.verbose > 0):
					print "\n\n\n\n\nselected variable thus far:\n%s" % "\n".join(temp_str)
					print "Ranking features iteration %02d" % (iteration_number)
				# Perform Single Input Single Output (SISO) for subsequent iterations.
				selected_variable,best_acc_t = self._siso(X, Y, iteration_number)
				if(self.verbose > 1):
					print "Evaluating MISO after iteration %02d" % (iteration_number)
				# Check if the feature has already been selected i.e. stopping condition 3 as mentioned above.
				#if selected_variable in self.all_selected_variables:
				if all(x in self.all_selected_variables for x in selected_variable):
					repeated_variable = True
				else:
					# The selected feature is stored inside self.all_selected_variables.
					for x in selected_variable:
						if x not in self.all_selected_variables:
							self.all_selected_variables.extend([x])

					# Perform Multiple Input Single Output (MISO) for subsequent iterations.
					acc_t_miso = self._miso(X[:, self.all_selected_variables], Y, iteration_number)
					# Accuracy of selected features is stored in accuracy_.
					self.accuracy_.append(acc_t_miso)
					# stop_epsilon makes sure the accuracy doesn't fall below the threshold i.e stopping condition 2 as mentioned above.
					stop_epsilon = self.accuracy_[iteration_number-1] - self.accuracy_[iteration_number-2]
					if(self.verbose > 1):
						print "::::::::::::::::::::accuracy of MISO after iteration %02d is %05f" % (iteration_number, acc_t_miso)
					iteration_number = iteration_number + 1
			# Stopping Condtion 1 -> Maximum number of features reached.
			if iteration_number > self.max_number_of_features:
				if(self.verbose > 0):
					print "Selection stopped: Maximum number of iteration %02d has been reached." % (self.max_number_of_features)
				self.stopping_condition_ = "max_number_of_features_reached"
				self.selected_subset_ = self.all_selected_variables
			# Stopping Condtion 2 -> epsilon value falls below the threshold.
			if stop_epsilon <= self.epsilon:
				if(self.verbose > 0):
					print "Selection stopped: Tolerance has been reached."
				print "Stopping Condition triggered at iteration number: %d" %(iteration_number-1)
				self.stopping_condition_ = "tolerance_reached"
				self.selected_subset_ = self.all_selected_variables[:-1]
				self.complete_subset_ = self.all_selected_variables[:-1]
				self.accuracy_ = self.accuracy_[:-1]
				print "Selected variables so far:"
				print self.selected_subset_
				index = 0
				while(len(self.complete_subset_) < self.max_number_of_features):
					if(self.all_ranking_[index] not in self.complete_subset_):
						self.complete_subset_.append(self.all_ranking_[index])
					index = index + 1
				print "The Complete Subset is:"
				print self.complete_subset_
			# Stopping Condtion 3 -> A specific feature has been already selected previously.
			if repeated_variable:
				if(self.verbose > 0):
					print "Selection stopped: A variable has been selected twice."
				print "Stopping Condition triggered at iteration number: %d" %(iteration_number-1)
				self.stopping_condition_ = "variable_selected_twice"
				self.selected_subset_ = self.all_selected_variables[:]
				self.complete_subset_ = self.all_selected_variables[:]
				self.accuracy_ = self.accuracy_
				print "Selected variables so far:"
				print self.selected_subset_
				print "Siso ranking at iteration number %d:"%(iteration_number-1)
				print self.all_ranking_
				index = 0
				while(len(self.complete_subset_) < self.max_number_of_features):
					if(self.all_ranking_[index] not in self.complete_subset_):
						self.complete_subset_.append(self.all_ranking_[index])
					index = index + 1
				print "The Complete Subset is:"
				print self.complete_subset_

	def _siso(self, X, Y, iteration_number):
		"""
		Determines which feature to select based on classification accuracy of
		the 'siso_ranking_size' ranked features from _input_ranking.
		"""
		warnings.filterwarnings("ignore")
		# Get a ranking of features based on the estimator.
		ranking, self.all_ranking_ = self._input_ranking(X, Y, iteration_number)
		self.siso_ranking_[(iteration_number-1), :] = ranking
		kf = KFold(n_splits=self.number_of_folds, shuffle=True,random_state=275)

		# combination of features from the ranking up to siso_order size 
		combs = []
		for i in range(self.siso_order):
			temp_comb = [list(x) for x in itertools.combinations(ranking, i+1)]
			combs.extend(temp_comb)
		acc_t_all = np.zeros((len(combs), 1))
		for idx_1, i in enumerate(combs):
			if(self.verbose > 1):
				print "...Evaluating SISO combination %02d which is %s" % (idx_1+1, str(i))
			X_temp = X[:, i]
			n = len(X_temp)
			X_temp = X_temp.reshape(n, len(i))
			count = 1
			acc_t_folds = np.zeros((self.number_of_folds, 1))
			# Compute accuracy for each SISO input.
			for train_index, test_index in kf.split(X_temp):
				X_train, X_test = X_temp[train_index], X_temp[test_index]
				y_train, y_test = Y[train_index], Y[test_index]
				self.estimator[1].fit(X_train, np.ravel(y_train), sample_weight=self.global_sample_weights[train_index])
				yHat_test = self.estimator[1].predict(X_test)
				acc_t = accuracy_score(y_test, yHat_test) #, sample_weight=self.global_sample_weights[test_index])
				if(self.verbose > 1):
					print "Fold %02d accuracy = %05f" % (count, acc_t)
				acc_t_folds[count-1, :] = acc_t
				count = count + 1
			acc_t_all[idx_1, :] = np.mean(acc_t_folds)
			if(self.verbose > 1):
				print "accuracy for combination %02d is = %05f" % (idx_1+1, np.mean(acc_t_folds))
		best_acc_t = np.amax(acc_t_all)
		# Feature with highest accuracy amongst siso ranked features is selected.
		#selected_variable = ranking[np.argmax(acc_t_all)]
		selected_variable = combs[np.argmax(acc_t_all)]
		if(self.verbose > 1):
			print "Selected variable is %s with accuracy %05f" % (str(selected_variable), best_acc_t)
		return selected_variable, best_acc_t

	def _miso(self, X, Y, iteration_number):
		"""
		Calculates the accuracy of selected features one additional feature at a
		time and also computes the updated weights of the samples.
		"""
		warnings.filterwarnings("ignore")
		kf = KFold(n_splits=self.number_of_folds, shuffle=True,random_state=275)
		count = 1
		acc_t_folds = np.zeros((self.number_of_folds, 1))
		# Compute the accuracy of the selected features one addition at a time.
		for train_index, test_index in kf.split(X):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Y[train_index], Y[test_index]
			self.estimator[2].fit(X_train, np.ravel(y_train), sample_weight=None)
			yHat_test = self.estimator[2].predict(X_test)
			acc_t = accuracy_score(y_test, yHat_test)
			if(self.verbose > 1):
				print "Fold %02d accuracy = %05f" % (count, acc_t)
			acc_t_folds[count-1,:] = acc_t
			count = count + 1
		acc_t_miso = np.mean(acc_t_folds)
		# Calculate the residual weights from fitting on the entire dataset.
		self.estimator[2].fit(X, np.ravel(Y), sample_weight=None)  #sample_weight=self.global_sample_weights)
		yHat_train_full = self.estimator[2].predict(X)
		if(self.loss == "adaboost"):
			# Determine the missclassified samples.
			acc_train_full = accuracy_score(Y, yHat_train_full, sample_weight=self.global_sample_weights)
			err = 1-acc_train_full
			alpha = np.log((1-err)/err) + np.log(self.n_classes_-1)
			misclass = np.subtract(Y.reshape(len(Y), 1), yHat_train_full.reshape(len(Y), 1))
			misclass_idx = np.nonzero(misclass)
			misclass_idx = misclass_idx[0]
			# Get the correct classification.
			corclass_idx = np.nonzero(misclass == 0)
			corclass_idx = corclass_idx[0]
			# Weighting up/down misclassified/classified samples.
			self.global_sample_weights[misclass_idx] = self.global_sample_weights[misclass_idx]*np.exp(alpha)
			if(iteration_number == 1):
				self.residual_weights_[(iteration_number-1), misclass_idx] = self.residual_weights_[(iteration_number-1), misclass_idx] + 1
			else:
				self.residual_weights_[(iteration_number-1), :] = self.residual_weights_[(iteration_number-2), :]
				self.residual_weights_[(iteration_number-1), misclass_idx] = self.residual_weights_[(iteration_number-1), misclass_idx] + 1

		elif(self.loss == "binary_crossentropy" or self.loss == "softmax"):
			# Determine the missclassified samples.
			if(iteration_number == 1):
				self.residual_weights_[(iteration_number-1), :] = 0
			else:
				self.residual_weights_[(iteration_number-1), :] = self.global_sample_weights
			# Gets all the labels.
			labels = np.unique(np.ravel(Y))
			Y_class = np.zeros((len(Y),len(labels)))
			prediction_probabiltiy = self.estimator[2].predict_proba(X)
			probability_weight = np.zeros(np.shape(Y))
			# Generates One-Hot encodings for Multi-Class Problems or Assigns 0/1
			#value for binary classification problems.
			for i in range(0, len(X)):
				for j in range(0, len(labels)):
					if(Y[i] == labels[j]):
						if(len(labels) == 2):
							Y[i] = j
							probability_weight[i] = prediction_probabiltiy[i][j]
						Y_class[i][j] = 1
			log_bias = 1e-30
			# Apply Binary Cross Entropy loss for Binary Classification Problems.
			if(self.loss == "binary_crossentropy"):
				alpha = -(Y * np.log(probability_weight+log_bias) + (1-np.array(Y))*np.log(1-probability_weight+log_bias))
			# Apply Softmax for Multi-Class Problems.
			elif(self.loss == "softmax"):
				alpha = -np.sum(Y_class*np.log(prediction_probabiltiy+log_bias), axis=1)
			self.global_sample_weights = self.global_sample_weights*alpha
			self.residual_weights_[(iteration_number-1), :] = self.global_sample_weights
		return acc_t_miso

	def _input_ranking(self, X, Y, iteration_number):
		"""
		Creates an initial ranking of features using the provided estimator for
		SISO evaluation.
		"""
		# Perform an initial ranking of features using the given estimator.
		check_estimator = str(self.estimator[0])
		if("XGBClassifier" in check_estimator):
			self.estimator[0].fit(X, np.ravel(Y), sample_weight=self.global_sample_weights)
			fscore = self.estimator[0].get_booster().get_score(importance_type='gain')
			feature_importance = np.zeros(X.shape[1])
			for k, v in fscore.iteritems():
				feature_importance[int(k[1:])] = v
			feature_rank = np.argsort(feature_importance)
			all_ranking = feature_rank[::-1]
			if(self.verbose > 1):
				print "feature importances of all available feature:"
			count = 0
			if type(self.siso_ranking_size) is int:
				for i in range(-1, -1*self.siso_ranking_size-1, -1):
					if(self.verbose > 1):
						print "%s   %05f" % (self.feature_names[feature_rank[i]], feature_importance[feature_rank[i]])
					count = count + 1
				# Return the 'siso_ranking_size' ranked features to perform SISO.
				return feature_rank[:-1*self.siso_ranking_size-1:-1], all_ranking

			elif type(self.siso_ranking_size) is list:
				assert len(self.siso_ranking_size) == 2, ("siso_ranking_size of list type is of incompatible format. Please enter a list of the following type: \n siso_ranking_size=[5, 10] \n Read documentation for more details.")
				for i in range(-1, -1*self.siso_ranking_size[1]-1, -1):
					if(self.verbose > 1):
						print "%s   %05f" % (self.feature_names[feature_rank[i]], feature_importance[feature_rank[i]])
					count = count + 1
				# Return the 'siso_ranking_size' ranked features to perform SISO.
				feature_rank = feature_rank[:-1*self.siso_ranking_size[1]-1:-1]
				return np.random.choice(feature_rank, self.siso_ranking_size[0], replace=False), all_ranking
		else:
			self.estimator[0].fit(X, np.ravel(Y), sample_weight=self.global_sample_weights)
			feature_importance = self.estimator[0].feature_importances_
			feature_rank = np.argsort(feature_importance)
			all_ranking = feature_rank[::-1]
			if(self.verbose > 1):
				print "feature importances of all available feature:"
			if type(self.siso_ranking_size) is int:
				for i in range(-1, -1*self.siso_ranking_size-1, -1):
					if(self.verbose > 1):
						print "%s   %05f" % (self.feature_names[feature_rank[i]], feature_importance[feature_rank[i]])
				return feature_rank[:-1*self.siso_ranking_size-1:-1], all_ranking
			elif type(self.siso_ranking_size) is list:
				assert len(self.siso_ranking_size) == 2, ("/* SISO CONDITION */")
				for i in range(-1, -1*self.siso_ranking_size[1]-1, -1):
					if(self.verbose > 1):
						print "%s   %05f" % (self.feature_names[feature_rank[i]], feature_importance[feature_rank[i]])
				feature_rank = feature_rank[:-1*self.siso_ranking_size[1]-1:-1]
				return np.random.choice(feature_rank, self.siso_ranking_size[0], replace=False), all_ranking

	def _transform(self, X):
		# Check if the fit(X, Y) function has been called prior to performing transform(X)
		try:
			self.selected_subset_
		except AttributeError:
			raise ValueError('fit(X, Y) needs to be called before using transform(X).')
		return X[:, self.selected_subset_]

	def _fit_transform(self, X, Y):
		self._fit(X, Y)
		return self._transform(X)
