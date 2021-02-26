import datajoint as dj
from functools import reduce

from nnfabrik.utility.nnf_helper import FabrikCache
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from .from_nnfabrik import TrainedModel, TrainedTransferModel
from .from_mei import Ensemble


DataCache = FabrikCache(base_table=Dataset, cache_size_limit=1)
TrainedModelCache = FabrikCache(base_table=TrainedModel, cache_size_limit=1)
TransferTrainedModelCache = FabrikCache(base_table=TrainedTransferModel, cache_size_limit=1)
EnsembleModelCache = FabrikCache(base_table=Ensemble, cache_size_limit=1)


def rename_table_attributes(tables, attrib, attrib_names, unit_table=False):
    if unit_table:
        proj_tables = [(dj.U(attrib_names[i]) * table.Units().proj(**{attrib_names[i]: attrib}))
                       for i, table in enumerate(tables)]
    else:
        proj_tables = [(dj.U(attrib_names[i])*table.proj(**{attrib_names[i]:attrib}))
                       for i, table in enumerate(tables)]

    joined_table = reduce(lambda x, y: x*y, proj_tables)
    return joined_table