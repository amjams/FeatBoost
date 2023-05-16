# FeatBoost
Python implementation of FeatBoost.

## Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eEySuIAJzmlNOChfLwEqJFKKbGNVYMwJ)

## Usage
```shell
pip install git+https://github.com/amjams/FeatBoost.git
```

Or just clone the repo (recommended for now)

```shell
git clone https://github.com/amjams/FeatBoost.git
```

```python
from featboost import FeatBoostClassifier

clf = FeatBoostClassifier()
clf.fit(X, y)
clf.feature_importances_
```
