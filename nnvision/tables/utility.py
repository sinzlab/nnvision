import datajoint as dj
from functools import reduce

from nnfabrik.utility.nnf_helper import FabrikCache
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from neuralpredictors.layers.readouts import PointPooled2d, FullGaussian2d

from .from_nnfabrik import TrainedModel, TrainedTransferModel
from .from_mei import Ensemble

DataCache = FabrikCache(base_table=Dataset, cache_size_limit=1)
TrainedModelCache = FabrikCache(base_table=TrainedModel, cache_size_limit=1)
TransferTrainedModelCache = FabrikCache(
    base_table=TrainedTransferModel, cache_size_limit=1
)
EnsembleModelCache = FabrikCache(base_table=Ensemble, cache_size_limit=1)


def rename_table_attributes(tables, attrib, attrib_names, unit_table=False):
    if unit_table:
        proj_tables = [
            (dj.U(attrib_names[i]) * table.Units().proj(**{attrib_names[i]: attrib}))
            for i, table in enumerate(tables)
        ]
    else:
        proj_tables = [
            (dj.U(attrib_names[i]) * table.proj(**{attrib_names[i]: attrib}))
            for i, table in enumerate(tables)
        ]

    joined_table = reduce(lambda x, y: x * y, proj_tables)
    return joined_table


def augment_ensemble_model(model, data_keys=None, unit_indices=None):
    if data_keys is None or unit_indices is None:
        raise ValueError("not implemented yet")

    if not hasattr(model, 'members'):
        raise ValueError("Model has to be an ensemble model with the 'members' property")

    for ensemble_member in model.members:
        data_key = list(ensemble_member.readout.keys())[0]
        in_shape = ensemble_member.readout[data_key].in_shape
        total_n_neurons = len(unit_indices)

        ensemble_member.readout['augmentation'] = FullGaussian2d(in_shape=in_shape,
                                                                 outdims=total_n_neurons,
                                                                 bias=True,
                                                                 gauss_type="isotropic")

        for i, (dk, idx) in enumerate(zip(data_keys, unit_indices)):
            features = ensemble_member.readout[dk].features.data[:, :, :, idx]
            bias = ensemble_member.readout[dk].bias.data[idx]
            sigma = ensemble_member.readout[dk].sigma.data[0][idx]
            mu = ensemble_member.readout[dk].mu.data[0][idx]
            ensemble_member.readout["augmentation"].features.data[:, :, :, i] = features
            ensemble_member.readout["augmentation"].bias.data[i] = bias
            ensemble_member.readout['augmentation'].sigma.data[:, i, :, :] = sigma
            ensemble_member.readout['augmentation'].mu.data[:, i, :, :] = mu

    return model

