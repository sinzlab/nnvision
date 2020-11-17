import datajoint as dj
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from .main import Recording
from ..utility.measures import get_oracles, get_repeats, get_FEV, get_explainable_var, get_correlations, get_poisson_loss, get_avg_correlations, get_predictions, get_targets
from .from_nnfabrik import TrainedModel, TrainedTransferModel
from .utility import DataCache, TrainedModelCache, EnsembleModelCache, TransferTrainedModelCache
from nnfabrik.utility.dj_helpers import CustomSchema
from nnfabrik.template import ScoringBase, SummaryScoringBase
from .from_nnfabrik import ScoringBaseNeuronType
from .from_mei import Ensemble

schema = CustomSchema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class TrainCorrelationScore(ScoringBaseNeuronType):
    trainedmodel_table = TrainedModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "train"
    measure_attribute = "train_correlation"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class ValidationCorrelationScore(ScoringBaseNeuronType):
    trainedmodel_table = TrainedModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "validation"
    measure_attribute = "validation_correlation"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class TestCorrelationScore(ScoringBaseNeuronType):
    trainedmodel_table = TrainedModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class CorrelationToAverageScore(ScoringBaseNeuronType):
    trainedmodel_table = TrainedModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_avg_correlations)
    measure_attribute = "avg_correlation"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class FEVeScore(ScoringBaseNeuronType):
    trainedmodel_table = TrainedModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_FEV)
    measure_dataset = "test"
    measure_attribute = "feve"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class TrainPoissonLoss(ScoringBaseNeuronType):
    trainedmodel_table = TrainedModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_poisson_loss)
    measure_dataset = "train"
    measure_attribute = "train_poissonloss"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class ValidationPoissonLoss(ScoringBaseNeuronType):
    trainedmodel_table = TrainedModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_poisson_loss)
    measure_dataset = "validation"
    measure_attribute = "validation_poissonloss"
    data_cache = DataCache
    model_cache = TrainedModelCache


@schema
class TestPoissonLoss(ScoringBaseNeuronType):
    trainedmodel_table = TrainedModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_poisson_loss)
    measure_dataset = "test"
    measure_attribute = "test_poissonloss"
    data_cache = DataCache
    model_cache = TrainedModelCache


# ============================= CUSTOM SCORES =============================


@schema
class TestPredictions(ScoringBaseNeuronType):
    trainedmodel_table = TrainedModel
    unit_table = Recording.Units
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
                if "unit_id" in key: key.pop("unit_id")
                if "data_key" in key: key.pop("data_key")
                if "unit_type" in key: key.pop("unit_type")
                neuron_key = dict(unit_index=unit_index, data_key=data_key)
                unit_type = ((self.unit_table & key) & neuron_key).fetch1("unit_type")
                unit_id = ((self.unit_table & key) & neuron_key).fetch1("unit_id")
                key["unit_id"] = unit_id
                key["unit_type"] = unit_type
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


# ============================= Ensemble SCORES =============================

@schema
class TestCorrelationScoreEnsemble(ScoringBaseNeuronType):
    trainedmodel_table = Ensemble
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache



@schema
class ValidationCorrelationScoreEnsemble(ScoringBaseNeuronType):
    trainedmodel_table = Ensemble
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "validation"
    measure_attribute = "validation_correlation"
    data_cache = DataCache
    model_cache = EnsembleModelCache


# ============================= SUMMARY SCORES =============================


@schema
class FEVe_thresholded(SummaryScoringBase):
    trainedmodel_table = TrainedModel
    measure_function = staticmethod(get_FEV)
    function_kwargs = dict(threshold=0.15, per_neuron=False)
    measure_dataset = "test"
    measure_attribute = "feve"
    data_cache = DataCache
    model_cache = TrainedModelCache

# ============================= TransferTrainedModel SCORES =============================


@schema
class TransferTestCorrelationScore(ScoringBaseNeuronType):
    trainedmodel_table = TrainedTransferModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test"
    measure_attribute = "test_correlation"
    data_cache = None
    model_cache = None


@schema
class TransferValidationCorrelationScore(ScoringBaseNeuronType):
    trainedmodel_table = TrainedTransferModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "validation"
    measure_attribute = "validation_correlation"
    data_cache = None
    model_cache = None


@schema
class TransferTestCorrelationMEICropped(ScoringBaseNeuronType):
    trainedmodel_table = TrainedTransferModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test_mei_cropped"
    measure_attribute = "test_correlation"
    data_cache = DataCache
    model_cache = TransferTrainedModelCache


@schema
class TransferTestCorrelationMEIUncropped(ScoringBaseNeuronType):
    trainedmodel_table = TrainedTransferModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test_mei_uncropped"
    measure_attribute = "test_correlation"
    data_cache = DataCache
    model_cache = TransferTrainedModelCache


@schema
class TransferTestCorrelationControlCropped(ScoringBaseNeuronType):
    trainedmodel_table = TrainedTransferModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test_control_cropped"
    measure_attribute = "test_correlation"
    data_cache = DataCache
    model_cache = TransferTrainedModelCache


@schema
class TransferTestCorrelationControlUncropped(ScoringBaseNeuronType):
    trainedmodel_table = TrainedTransferModel
    unit_table = Recording.Units
    measure_function = staticmethod(get_correlations)
    measure_dataset = "test_control_uncropped"
    measure_attribute = "test_correlation"
    data_cache = DataCache
    model_cache = TransferTrainedModelCache
