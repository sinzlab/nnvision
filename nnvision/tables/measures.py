import datajoint as dj
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnvision.tables.legacy.main import MonkeyExperiment

from ..utility.dj_helpers import get_default_args
from ..utility.measures import get_oracles, get_repeats, get_FEV, get_explainable_var, get_correlations, get_poisson_loss, \
    get_avg_correlations, get_oracles_corrected, get_model_rf_size, get_avg_firing
from .utility import DataCache
from .from_nnfabrik import MeasuresBaseNeuronType
from .main import Recording
from .templates import SummaryMeasuresBase
from nnfabrik.utility.dj_helpers import CustomSchema
from nnfabrik.builder import resolve_model

schema = CustomSchema(dj.config.get('nnfabrik.schema_name', 'nnfabrik_core'))


@schema
class ExplainableVar(MeasuresBaseNeuronType):
    dataset_table = Dataset
    unit_table = Recording.Units
    measure_function = staticmethod(get_explainable_var)
    measure_dataset = "test"
    measure_attribute = "fev"
    data_cache = DataCache


@schema
class AvgFiringTest(MeasuresBaseNeuronType):
    dataset_table = Dataset
    unit_table = Recording.Units
    measure_function = staticmethod(get_avg_firing)
    measure_dataset = "test"
    measure_attribute = "test_avg_firing"
    data_cache = DataCache


@schema
class OracleScore(MeasuresBaseNeuronType):
    dataset_table = Dataset
    unit_table = Recording.Units
    measure_function = staticmethod(get_oracles)
    measure_dataset = "test"
    measure_attribute = "oracle_score"
    data_cache = DataCache

@schema
class OracleScoreCorrected(MeasuresBaseNeuronType):
    dataset_table = Dataset
    unit_table = Recording.Units
    measure_function = staticmethod(get_oracles_corrected)
    measure_dataset = "test"
    measure_attribute = "oracle_score_corrected"
    data_cache = DataCache


@schema
class ModelRFSize(SummaryMeasuresBase):
    dataset_table = Model
    unit_table = None
    measure_function = staticmethod(get_model_rf_size)
    measure_attribute = "model_rf_size"

    def make(self, key):
        model_config = (self.dataset_table & key).fetch1("model_config")
        default_args = get_default_args(resolve_model((self.dataset_table & key).fetch1("model_fn")))
        default_args.update(model_config)
        key[self.measure_attribute] = self.measure_function(default_args)
        self.insert1(key, ignore_extra_fields=True)
