import numpy as np
import pytest
from featboost import FeatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier


@pytest.fixture
def X():
    return np.array([[1, 2, 3], [-4, -5, -6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])


@pytest.fixture
def y():
    return np.array([0, 1, 0, 1, 0])


def test_initialization():
    """Should be able to initialize without changing default params."""
    clf = ExtraTreesClassifier()
    fs = FeatBoostClassifier(estimator=clf)
    assert fs is not None


def test_fit(X, y):
    """Should be able to fit by setting `siso_ranking_size` to amount of dimensions p."""
    n, p = X.shape

    clf = ExtraTreesClassifier()
    fs = FeatBoostClassifier(estimator=clf, siso_ranking_size=p, number_of_folds=5)
    fs.fit(X, y)

    assert fs.selected_subset_ is not None
    assert fs.feature_importances_array_ is not None
