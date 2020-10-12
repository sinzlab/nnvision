from nnfabrik.utility.nnf_helper import FabrikCache
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from .from_nnfabrik import TrainedModel
from .from_mei import Ensemble


DataCache = FabrikCache(base_table=Dataset, cache_size_limit=1)
TrainedModelCache = FabrikCache(base_table=TrainedModel, cache_size_limit=1)
EnsembleModelCache = FabrikCache(base_table=Ensemble, cache_size_limit=1)
