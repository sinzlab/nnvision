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
from nnfabrik.template import ScoringBase
from ..tables.from_nnfabrik import TrainedModel

schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class TrainCorrelation(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    scoring_function = staticmethod(get_correlations)
    scoring_dataset = "train"
    scoring_attribute = "train_correlation"


@schema
class ValidationCorrelation(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    scoring_function = staticmethod(get_correlations)
    scoring_dataset = "validation"
    scoring_attribute = "validation_correlation"


@schema
class TestCorrelation(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    scoring_function = staticmethod(get_correlations)
    scoring_dataset = "test"
    scoring_attribute = "test_correlation"


@schema
class AverageCorrelation(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    scoring_function = staticmethod(get_avg_correlations)
    scoring_attribute = "avg_correlation"


@schema
class FEVe(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    scoring_function = staticmethod(get_FEV)
    scoring_dataset = "test"
    scoring_attribute = "feve"


@schema
class TrainPoissonLoss(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    scoring_function = staticmethod(get_poisson_loss)
    scoring_dataset = "train"
    scoring_attribute = "test_poissonloss"


@schema
class ValidationPoissonLoss(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    scoring_function = staticmethod(get_poisson_loss)
    scoring_dataset = "validation"
    scoring_attribute = "test_poissonloss"


@schema
class TestPoissonLoss(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    scoring_function = staticmethod(get_poisson_loss)
    scoring_dataset = "test"
    scoring_attribute = "test_poissonloss"


