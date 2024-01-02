import datajoint as dj
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnvision.tables.legacy.main import MonkeyExperiment

from ..utility.dj_helpers import get_default_args
from ..utility.measures import (
    get_oracles,
    get_repeats,
    get_FEV,
    get_explainable_var,
    get_correlations,
    get_poisson_loss,
    get_avg_correlations,
    get_oracles_corrected,
    get_model_rf_size,
    get_avg_firing,
)
from .utility import DataCache
from .from_nnfabrik import MeasuresBaseNeuronType
from .main import Recording
from .templates import SummaryMeasuresBase
from nnfabrik.utility.dj_helpers import CustomSchema
from nnfabrik.builder import resolve_model

schema = CustomSchema(dj.config.get("nnfabrik.schema_name", "nnfabrik_core"))


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


