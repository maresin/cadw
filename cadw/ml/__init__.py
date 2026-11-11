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
from .lof import LOFResult, fit_lof
from .kernel_pca import (KernelPCAResult, fit_kernel_pca)
from .nystroem import (NystroemResult, fit_nystroem_kernel_map)
from .spectral_clustering import (SpectralClusteringResult, fit_spectral_clustering)
from .truncated_svd import (TruncatedSVDResult, fit_truncated_svd)
from .spectral_coclustering import (SpectralCoclusteringResult, fit_spectral_coclustering)
from .knn_imputer import (KNNImputerResult, fit_knn_imputer)
from .naive_bayes import (
    GaussianNB, fit_gaussian_nb,
    MultinomialNB, fit_multinomial_nb,
    BernoulliNB, fit_bernoulli_nb,
)

from cadw.ml.stratified_split import (  # noqa: E402
    StratifiedSplitModel, fit_stratified_split,
)
from cadw.ml.one_hot_encoder import (  # noqa: E402
    OneHotEncoderModel, fit_one_hot_encoder,
)
from cadw.ml.reservoir_sample import (  # noqa: E402
    ReservoirSampleModel, fit_reservoir_sample,
)
from cadw.ml.kernel_density import (  # noqa: E402
    KernelDensityModel, fit_kernel_density,
)
from cadw.ml.complement_nb import (  # noqa: E402
    ComplementNBModel, fit_complement_nb,
)
from cadw.ml.weibull_aft import WeibullAFTModel, fit_weibull_aft  # noqa: E402
from cadw.ml.kaplan_meier import KaplanMeierModel, fit_kaplan_meier  # noqa: E402
from cadw.ml.nmf import NMFModel, fit_nmf  # noqa: E402
from cadw.ml.bandits import (
    EpsilonGreedyPolicy, UCB1Policy, ThompsonSamplingPolicy,
    fit_epsilon_greedy, fit_ucb1, fit_thompson_sampling,
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
    "fit_epsilon_greedy",
    "fit_ucb1",
    "fit_thompson_sampling",
    "EpsilonGreedyPolicy",
    "UCB1Policy",
    "ThompsonSamplingPolicy",
    "LOFResult", "fit_lof",
    "KNNImputerResult", "fit_knn_imputer",
    "KernelPCAResult", "fit_kernel_pca",
    "NystroemResult", "fit_nystroem_kernel_map",
    "SpectralClusteringResult", "fit_spectral_clustering",
    "TruncatedSVDResult", "fit_truncated_svd",
    "SpectralCoclusteringResult", "fit_spectral_coclustering",
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
# v0.9.61: Autoencoder MLP (reconstruction-based DL)
from .autoencoder import AutoencoderResult, fit_autoencoder  # noqa: E402,F401
# v0.9.62: KMeansAD (k-means clustering on windows)
from .kmeans_ad import KMeansADResult, fit_kmeans_ad  # noqa: E402,F401
# v0.9.63: PatchMemoryAD (PaAno-inspired patch+memory+triplet)
from .patch_memory_ad import (  # noqa: E402,F401
    PatchMemoryADResult, fit_patch_memory_ad,
)
# v0.9.70: BinarySegmentation (greedy top-down L2 splitting)
from .binseg import BinSegResult, fit_binseg  # noqa: E402,F401
# v0.9.72: BOCPD (Bayesian online change-point, Student-t)
from .bocpd import BOCPDResult, fit_bocpd  # noqa: E402,F401
# v0.9.84: AnomalyPipeline (contextual/structural)
from .anomaly_pipeline import (  # noqa: E402,F401
    AnomalyPipeline,
    AnomalyPipelineResult,
)

from cadw.ml.rocket import fit_rocket, RocketModel  # noqa: E402

# v0.9.166 (DEC-061): stacking infrastructure
from cadw.ml.stacking import fit_stacking, StackingModel
