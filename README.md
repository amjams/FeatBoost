# FeatBoost
Python implementation of FeatBoost.

## Usage
```shell
pip install git+https://github.com/amjams/FeatBoost.git
```

```python
from featboost import FeatBoostClassifier

clf = FeatBoostClassifier()
clf.fit(X, y)
clf.feature_importances_
```
