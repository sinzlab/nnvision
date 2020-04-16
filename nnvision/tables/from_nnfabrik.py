import datajoint as dj
from nnfabrik.template import TrainedModelBase
import tempfile
import torch
import os
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.utility.dj_helpers import gitlog, make_hash
import numpy as np
from .main import MonkeyExperiment
from ..utility.measures import get_oracles, get_repeats, get_FEV, get_explainable_var, get_correlations, get_poisson_loss, get_avg_correlations

schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class TrainedModel(TrainedModelBase):
    table_comment = "Trained models"
