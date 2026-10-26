"""CADW ML layer (v0.7.6)."""
from .linear import (
    XTXOp, LinearModel, fit_linear,
    fit_lasso, fit_elasticnet,
)
from .logistic import LogisticGradOp, LogisticModel, fit_logistic
from .kmeans import KMeansStepOp, KMeansModel, fit_kmeans
from .pca import PCA, fit_pca
from .gbdt import (
    GBDTRegressor, fit_gbdt_regressor,
    GBDTClassifier, fit_gbdt_classifier,
    GBDTMulticlassClassifier, fit_gbdt_multiclass,
)
from .gmm import GMMModel, fit_gmm
from .importance import permutation_importance
from .iforest import IsolationForestModel, fit_isolation_forest
from .naive_bayes import (
    GaussianNB, fit_gaussian_nb,
    MultinomialNB, fit_multinomial_nb,
    BernoulliNB, fit_bernoulli_nb,
)

__all__ = [
    "XTXOp", "LinearModel", "fit_linear",
    "fit_lasso", "fit_elasticnet",
    "LogisticGradOp", "LogisticModel", "fit_logistic",
    "KMeansStepOp", "KMeansModel", "fit_kmeans",
    "PCA", "fit_pca",
    "GBDTRegressor", "fit_gbdt_regressor",
    "GBDTClassifier", "fit_gbdt_classifier",
    "GBDTMulticlassClassifier", "fit_gbdt_multiclass",
    "GaussianNB", "fit_gaussian_nb",
    "MultinomialNB", "fit_multinomial_nb",
    "BernoulliNB", "fit_bernoulli_nb",
    "IsolationForestModel", "fit_isolation_forest",
    "permutation_importance",
    "GMMModel", "fit_gmm",
]
# v0.9.52: DWT-MLEAD (Thill, Konen, Baeck 2017)
from .dwt_mlead import DwtMleadResult, fit_dwt_mlead  # noqa: E402,F401
# v0.9.53: KShapeAD (Paparrizos & Gravano 2015)
from .kshape import KShapeResult, fit_kshape  # noqa: E402,F401
# v0.9.54: Sub-PCA (windowed PCA reconstruction error)
from .sub_pca import SubPCAResult, fit_sub_pca  # noqa: E402,F401
# v0.9.55: Extended Isolation Forest (Hariri et al. 2019)
from .eif import EIFResult, fit_eif  # noqa: E402,F401
# v0.9.56: T-squared + Q (Hotelling SPC on PCA)
from .tspq import T2QResult, fit_t2q  # noqa: E402,F401
# v0.9.57: Hampel filter + ZMS (rolling baselines)
from .hampel import HampelResult, fit_hampel  # noqa: E402,F401
from .zms import ZMSResult, fit_zms  # noqa: E402,F401
# v0.9.58: CUSUM (cumulative sum control chart)
from .cusum import CUSUMResult, fit_cusum  # noqa: E402,F401
# v0.9.59: STRAY (Talagala et al. 2021)
from .stray import STRAYResult, fit_stray  # noqa: E402,F401
# v0.9.60: S2G (Series2Graph, Boniol & Palpanas 2020)
from .s2g import S2GResult, fit_s2g  # noqa: E402,F401
