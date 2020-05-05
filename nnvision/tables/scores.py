import datajoint as dj
from nnfabrik.template import TrainedModelBase
import tempfile
import torch
import os
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.utility.dj_helpers import gitlog, make_hash
import numpy as np
from .main import MonkeyExperiment
from ..utility.measures import get_oracles, get_repeats, get_FEV, get_explainable_var, get_correlations, get_poisson_loss, get_avg_correlations, get_predictions, get_targets
from .from_nnfabrik import TrainedModel
from .from_mei import TrainedEnsembleModel
from .utility import DataCache, TrainedModelCache, EnsembleModelCache
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


@schema
class TrainCorrelationEnsemble(ScoringBase):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MonkeyExperiment.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "train"
    measure_attribute = "train_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache


@schema
class ValidationCorrelationEnsemble(ScoringBase):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MonkeyExperiment.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "validation"
    measure_attribute = "validation_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache


@schema
class TestCorrelationEnsemble(ScoringBase):
    trainedmodel_table = TrainedEnsembleModel
    dataset_table = Dataset
    unit_table = MonkeyExperiment.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache


@schema
class TestPredictions(ScoringBase):
    trainedmodel_table = TrainedModel
    unit_table = MonkeyExperiment.Units
    measure_function = staticmethod(get_predictions)
    measure_secondary_function = staticmethod(get_targets)
    measure_dataset = "test"
    measure_attribute = "test_predictions"
    measure_secondary_attribute = "test_responses"
    data_cache = DataCache
    model_cache = TrainedModelCache
    measure_function_kwargs = dict(test_data=True)

    # table level comment
    table_comment = "A template table for storing results/scores of a TrainedModel"

    @property
    def definition(self):
        definition = """
                # {table_comment}
                -> self.trainedmodel_table
                ---
                {measure_attribute}:      longblob     # A template for a computed score of a trained model
                {measure_secondaty_attribute}:      longblob     # A template for a computed score of a trained model
                {measure_attribute}_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
                """.format(table_comment=self.table_comment, measure_attribute=self.measure_attribute, measure_secondaty_attribute=self.measure_secondary_attribute)
        return definition

    class Units(dj.Part):
        @property
        def definition(self):
            definition = """
                # Scores for Individual Neurons
                -> master
                -> master.unit_table
                ---
                unit_{measure_attribute}:     longblob   # A template for a computed unit score   
                unit_{measure_secondaty_attribute}:     longblob   # A template for a computed unit score     
                """.format(measure_attribute=self._master.measure_attribute, measure_secondaty_attribute=self._master.measure_secondary_attribute)
            return definition

    def make(self, key):
        dataloaders = self.get_repeats_dataloaders(key=key) if self.measure_dataset == 'test' else self.get_dataloaders(
            key=key)

        model = self.get_model(key=key)

        unit_predictions_dict = self.measure_function(model=model,
                                                   dataloaders=dataloaders,
                                                   device='cuda',
                                                   as_dict=True,
                                                   per_neuron=True,
                                                   **self.measure_function_kwargs)

        unit_targets_dict = self.measure_secondary_function(model=model,
                                                      dataloaders=dataloaders,
                                                      device='cuda',
                                                      as_dict=True,
                                                      per_neuron=True,
                                                      **self.measure_function_kwargs)

        key[self.measure_attribute] = unit_predictions_dict
        key[self.measure_secondary_attribute] = unit_targets_dict
        self.insert1(key, ignore_extra_fields=True)

        for data_key, unit_scores in unit_predictions_dict.items():
            for unit_index, unit_score in enumerate(unit_scores):
                unit_secondary_score = unit_targets_dict[data_key][unit_index]
                key.pop("unit_id") if "unit_id" in key else None
                key.pop("data_key") if "data_key" in key else None
                neuron_key = dict(unit_index=unit_index, data_key=data_key)
                unit_id = ((self.unit_table & key) & neuron_key).fetch1("unit_id")
                key["unit_id"] = unit_id
                key["unit_{}".format(self.measure_attribute)] = unit_score
                key["unit_{}".format(self.measure_secondary_attribute)] = unit_secondary_score
                key["data_key"] = data_key
                self.Units.insert1(key, ignore_extra_fields=True)


@schema
class ValidationPredictions(TestPredictions):
    measure_dataset = "validation"
    measure_attribute = "validation_predictions"
    measure_secondary_attribute = "validation_targets"
    measure_function_kwargs = dict(test_data=False)