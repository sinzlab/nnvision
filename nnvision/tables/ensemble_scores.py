import datajoint as dj
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.utility.nnf_helper import FabrikCache
from nnfabrik.utility.dj_helpers import CustomSchema
from nnfabrik.template import ScoringBase, SummaryScoringBase
from .main import Recording
from ..utility.measures import get_oracles, get_repeats, get_FEV, get_explainable_var, get_correlations, get_poisson_loss, get_avg_correlations, get_predictions, get_targets
from .from_nnfabrik import TrainedModel, TrainedTransferModel
from .from_mei import Ensemble
from .utility import DataCache
from .from_nnfabrik import ScoringBaseNeuronType
from .legacy.from_mei import TrainedEnsembleModel

EnsembleModelCache = FabrikCache(base_table=Ensemble, cache_size_limit=1)
EnsembleModelCache_legacy = FabrikCache(base_table=TrainedEnsembleModel, cache_size_limit=1)
schema = CustomSchema(dj.config.get('nnfabrik.schema_name', 'nnfabrik_core'))



@schema
class TrainCorrelationEnsembleScore(ScoringBaseNeuronType):
    trainedmodel_table = Ensemble
    dataset_table = Dataset
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "train"
    measure_attribute = "train_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache


@schema
class ValidationCorrelationEnsembleScore(ScoringBaseNeuronType):
    trainedmodel_table = Ensemble
    dataset_table = Dataset
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "validation"
    measure_attribute = "validation_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache


@schema
class TestCorrelationEnsembleScore(ScoringBaseNeuronType):
    trainedmodel_table = Ensemble
    dataset_table = Dataset
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache


@schema
class CorrelationToAverageEnsembleScore(ScoringBaseNeuronType):
    trainedmodel_table = Ensemble
    unit_table = Recording.Units
    measure_function = staticmethod(get_avg_correlations)
    measure_attribute = "avg_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache