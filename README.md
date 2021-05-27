# FeatBoost
Python implementation of FeatBoost. This module is not fully documented, as it is under review. 
To request a demo script, please contact Ahmad Alsahaf: a.m.j.a.alsahaf@rug.nl

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