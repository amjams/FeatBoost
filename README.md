# FeatBoost
This project showcases the Python implementation of FeatBoost for classification problems.

# Dependencies
• numpy

• scikit-learn

# Installation
Install with pip

# Usage
Download FeatBoost and use it just as any other scikit-learn code.

• fit(X, y)

• transform(X)

• fit_transform(X, y)

# Examples
```
from FeatBoost import FeatBoostClassification as FBC

samples = 1000
number_of_features = 40
informative_features = 5
redundant_features = 5
repeated_features = 0
class_separation = 0.2
number_of_classes = 2

X, y = make_classification(n_samples=samples, n_features=number_of_features, n_informative=informative_features, n_redundant=redundant_features, n_repeated=repeated_features, n_classes=number_of_classes, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=class_separation, hypercube=True, shift=0.0, scale=1.4, shuffle=False, random_state=None)

fbc = FBC(estimator = [XGB,knn,XGB], number_of_folds=10,
         epsilon=1e-18, loss='softmax', fast_mode=False, siso_ranking_size = 10,
         siso_order = 1, max_number_of_features = 100, metric='acc',
         xgb_importance ='gain', reset=True, verbose=0)

fbc.fit(X=X, Y=y)
```
