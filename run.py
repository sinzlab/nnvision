import datajoint as dj

dj.config["enable_python_native_blobs"] = True
dj.config["nnfabrik.schema_name"]= 'nnfabrik_toy_V4'
schema = dj.schema('nnfabrik_toy_V4')

from nnfabrik.main import *
import nnfabrik
from nnfabrik import main, builder
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

key_remapped = {'model_fn': 'nnvision.models.models.se_core_remapped_gauss_readout',
 'model_hash': 'fcf83704246f464b2e071135cc039dd7',
 'dataset_fn': 'nnvision.datasets.monkey_mua_sua_loader',
 'dataset_hash': 'ca53b3ae60291b7d55edd85b2a3b67ec',
 'trainer_fn': 'nnvision.training.trainers.nnvision_trainer',
 'trainer_hash': '7eba3d5e8d426d6bbcd3f248565f8cfb',
 'seed': 1000}

key_attention = {'model_fn': 'nnvision.models.models.se_core_attention_readout',
 'model_hash': '944ca98fa137d6edb15747d7517b9845',
 'dataset_fn': 'nnvision.datasets.monkey_mua_sua_loader',
 'dataset_hash': 'ca53b3ae60291b7d55edd85b2a3b67ec',
 'trainer_fn': 'nnvision.training.trainers.nnvision_trainer',
 'trainer_hash': '6c42dbd455b5b5a4a7eb37c8c91aecfa',
 'seed': 2000}

keys = [key_remapped, key_attention]
from nnvision.tables.from_nnfabrik import TrainedModel

TrainedModel.populate(keys, reserve_jobs=True)



#
import datajoint as dj

dj.config["enable_python_native_blobs"] = True
dj.config["nnfabrik.schema_name"]= 'nnfabrik_schema'
schema = dj.schema('nnfabrik_schema')
from nnvision.tables.from_nnfabrik import TrainedModel

keys = {}
TrainedModel.populate(keys, display_progress=True, reserve_jobs=True)