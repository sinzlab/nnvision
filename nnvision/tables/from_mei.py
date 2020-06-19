from __future__ import annotations

import datajoint as dj
from nnfabrik.main import Dataset
from .from_nnfabrik import TrainedModel
from mei import mixins
from mei.main import MEITemplate, MEISeed
from torch.utils.data import DataLoader
from nnfabrik.utility.dj_helpers import CustomSchema
from .main import Recording

from typing import Callable, Iterable, Mapping, Optional, Tuple, Dict, Any
Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

schema = CustomSchema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class Method(mixins.MEIMethodMixin, dj.Lookup):
    seed_table = MEISeed


@schema
class Ensemble(mixins.TrainedEnsembleModelTemplateMixin, dj.Manual):
    dataset_table = Dataset
    trained_model_table = TrainedModel
    class Member(mixins.TrainedEnsembleModelTemplateMixin.Member, dj.Part):
        pass


@schema
class MEI(mixins.MEITemplateMixin, dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    trained_model_table = Ensemble
    selector_table = Recording.Units
    method_table = Method
    seed_table = MEISeed