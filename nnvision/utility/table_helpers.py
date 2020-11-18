import numpy as np

from ..tables.from_nnfabrik import Model, TrainedModel, SharedReadoutTrainedModel
from ..tables.from_mei import SharedReadoutTrainedEnsembleModel


def fill_trained_model_table():
    pass


def create_ensembles():
    pass


def fill_augmented_model_tables(trainedmodel_keys,
                                model_fn=None,
                                model_config=None,
                                model_table=None,
                                trainedmodel_table=None,
                                transfer_trainer_hash='d41d8cd98f00b204e9800998ecf8427e',
                                transfer_seed=1000,
                                transfer_trainedmodel_table=None,
                                add_ensemble=True,
                                transfer_ensemble_table=None):
    """

    Args:
        trainedmodel_keys:
        model_fn:
        model_config:
        model_table:
        trainedmodel_table:
        transfer_trainer_hash:
        transfer_seed:
        transfer_trainedmodel_table:
        add_ensemble:
        transfer_ensemble_table:

    Returns: keys
    """


    # first check that the trainedmodel_keys only differ in seed, all other attributes have to be the same.
    for attribute in ["dataset_hash", "model_hash", "trainer_hash"]:
        if len(np.unique([key[attribute] for key in trainedmodel_keys])) != 1:
            raise ValueError(f"trainedmodel_keys are only allowed to differ in seed. "
                             f"{attribute} has more than one unique value")

    # setting up default args
    if model_fn is None:
        model_fn = 'nnvision.models.models.augmented_full_readout'

    if model_config is None:
        model_config = {'mua_in': False,
                        'n_augment_x': 8,
                        'n_augment_y': 8,
                         }

    if model_table is None:
        model_table = Model

    if trainedmodel_table is None:
        trainedmodel_table = TrainedModel

    if transfer_trainedmodel_table is None:
        transfer_trainedmodel_table = SharedReadoutTrainedModel

    if transfer_ensemble_table is None:
        transfer_ensemble_table = SharedReadoutTrainedEnsembleModel


    # setting up the return dictionary
    return_keys = {}

    # Add Augmented model to Model table
    print(f"adding entries to {model_table.__name__} table ...")
    inserted_key_model_hashes = []
    inserted_keys = []
    for key in trainedmodel_keys:
        n_augment_x = model_config["n_augment_x"]
        n_augment_y = model_config["n_augment_y"]
        model_config["key"] = key
        seed = key["seed"]
        model_comment = f"model_hash={key['model_hash']}, augmentation={n_augment_x}x{n_augment_y}, seed={seed}"

        insert_key = model_table().add_entry(model_fn=model_fn,
                                model_config=model_config,
                                model_comment=model_comment)
        inserted_key_model_hashes.append(insert_key["model_hash"])
        inserted_keys.append(insert_key)
    return_keys["model_insert_keys"] = inserted_keys

    # Fill transfer_trainedmodel_table with the recently added models
    print(f"populating {transfer_trainedmodel_table.__name__} table ...")

    dataset_hash = trainedmodel_keys[0]["dataset_hash"]
    transfer_trainedmodel_table_keys = []
    for model_hash in inserted_key_model_hashes:
        key = dict(model_hash=model_hash,
                   dataset_hash=dataset_hash,
                   trainer_hash=transfer_trainer_hash,
                   seed=transfer_seed)
        transfer_trainedmodel_table_keys.append(key)

    transfer_trainedmodel_table().populate(transfer_trainedmodel_table_keys, display_progress=True, reserve_jobs=True)
    return_keys["transfer_trainedmodel_table_keys"] = transfer_trainedmodel_table_keys

    if add_ensemble:
        print(f"filling {transfer_ensemble_table.__name__} table ...")
        transfer_ensemble_table().create_ensemble(key=transfer_trainedmodel_table_keys,
                                                  comment=model_comment)

        ensemble_hash = (transfer_ensemble_table.Member() & transfer_trainedmodel_table_keys).fetch("ensemble_hash", limit=1)[0]
        return_keys["transfer_ensemble_hash"] = ensemble_hash

    return return_keys




