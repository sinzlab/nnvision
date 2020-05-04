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
from ..tables.from_nnfabrik import TrainedModel
from .utility import DataCache, TrainedModelCache
from nnfabrik.utility.dj_helpers import CustomSchema
from nnfabrik.template import ScoringBase

schema = CustomSchema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class TrainCorrelation(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "train"
    measure_attribute = "train_correlation"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class ValidationCorrelation(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "validation"
    measure_attribute = "validation_correlation"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class TestCorrelation(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class AverageCorrelation(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    measure_function = staticmethod(get_avg_correlations)
    measure_attribute = "avg_correlation"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class FEVe(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    measure_function = staticmethod(get_FEV)
    measure_dataset = "test"
    measure_attribute = "feve"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class TrainPoissonLoss(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    measure_function = staticmethod(get_poisson_loss)
    measure_dataset = "train"
    measure_attribute = "train_poissonloss"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class ValidationPoissonLoss(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    measure_function = staticmethod(get_poisson_loss)
    measure_dataset = "validation"
    measure_attribute = "validation_poissonloss"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class TestPoissonLoss(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    measure_function = staticmethod(get_poisson_loss)
    measure_dataset = "test"
    measure_attribute = "test_poissonloss"
    data_cache = DataCache
    model_cache = TrainedModelCache
