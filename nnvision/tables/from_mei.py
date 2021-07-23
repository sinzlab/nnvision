from __future__ import annotations

import datajoint as dj
import warnings
from functools import partial

from nnfabrik.main import Dataset
from nnfabrik.utility.dj_helpers import CustomSchema, cleanup_numpy_scalar, make_hash
from nnfabrik.builder import resolve_fn

from .from_nnfabrik import TrainedModel, SharedReadoutTrainedModel, TrainedTransferModel
from .main import Recording

from mei import mixins
from mei.main import MEITemplate, MEISeed
from mei.modules import ConstrainedOutputModel
from torch.utils.data import DataLoader
from torch.nn import Module, ModuleList

from typing import Dict, Any

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

schema = CustomSchema(dj.config.get("schema_name", "nnfabrik_core"))
resolve_target_fn = partial(resolve_fn, default_base="targets")


@schema
class Method(mixins.MEIMethodMixin, dj.Lookup):
    seed_table = MEISeed


@schema
class MethodGroup(mixins.MEIMethodMixin, dj.Lookup):
    seed_table = MEISeed


@schema
class Ensemble(mixins.TrainedEnsembleModelTemplateMixin, dj.Manual):
    dataset_table = Dataset
    trained_model_table = TrainedModel

    class Member(mixins.TrainedEnsembleModelTemplateMixin.Member, dj.Part):
        pass


@schema
class SharedReadoutTrainedEnsembleModel(
    mixins.TrainedEnsembleModelTemplateMixin, dj.Manual
):
    dataset_table = Dataset
    trained_model_table = SharedReadoutTrainedModel

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


@schema
class MEIShared(mixins.MEITemplateMixin, dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    trained_model_table = SharedReadoutTrainedEnsembleModel
    selector_table = Recording.Units
    method_table = Method
    seed_table = MEISeed


@schema
class MEITargetFunctions(dj.Manual):
    definition = """
    target_fn:       varchar(64)
    target_hash:     varchar(64)
    ---
    target_config:   longblob
    target_comment:  varchar(128)
    """

    resolve_fn = resolve_target_fn

    @property
    def fn_config(self):
        target_fn, target_config = self.fetch1("target_fn", "target_config")
        target_config = cleanup_numpy_scalar(target_config)
        return target_fn, target_config

    def add_entry(
        self, target_fn, target_config, target_comment="", skip_duplicates=False
    ):
        """
        Add a new entry to the TargetFunction table.

        Args:
            target_fn (string) - name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
                dynamic name resolution will be attempted. Other wise, the name will be checked inside `targets` subpackage.
            target_config (dict) - Python dictionary containing keyword arguments for the target_fn
            dataset_comment - Optional comment for the entry.
            target_comment - If True, no error is thrown when a duplicate entry (i.e. entry with same target_fn and target_config) is found.

        Returns:
            key - key in the table corresponding to the new (or possibly existing, if skip_duplicates=True) entry.
        """

        try:
            resolve_target_fn(target_fn)
        except (NameError, TypeError) as e:
            warnings.warn(str(e) + "\nTable entry rejected")
            return

        target_hash = make_hash(target_config)
        key = dict(
            target_fn=target_fn,
            target_hash=target_hash,
            target_config=target_config,
            target_comment=target_comment,
        )

        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        return key

    def get_target_fn(self, key=None):
        if key is None:
            key = self.fetch("KEY")
        target_fn, target_config = (self & key).fn_config
        return partial(self.resolve_fn(target_fn), **target_config)


@schema
class MEITargetUnits(dj.Manual):
    definition = """
    unit_hash:       varchar(64)
    ---
    unit_ids:          longblob
    data_key:          varchar(64)
    unit_comment:      varchar(128)
    """

    def add_entry(
        self, unit_ids, data_key=None, unit_comment="", skip_duplicates=False
    ):
        """
        Add a new entry to the TargetFunction table.

        Args:
            target_fn (string) - name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
                dynamic name resolution will be attempted. Other wise, the name will be checked inside `targets` subpackage.
            target_config (dict) - Python dictionary containing keyword arguments for the target_fn
            dataset_comment - Optional comment for the entry.
            target_comment - If True, no error is thrown when a duplicate entry (i.e. entry with same target_fn and target_config) is found.

        Returns:
            key - key in the table corresponding to the new (or possibly existing, if skip_duplicates=True) entry.
        """

        unit_hash = make_hash([unit_ids, data_key])
        key = dict(
            unit_hash=unit_hash,
            unit_ids=unit_ids,
            data_key=data_key,
            unit_comment=unit_comment,
        )

        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        return key


@schema
class MEIObjective(dj.Computed):
    target_fn_table = MEITargetFunctions
    target_unit_table = MEITargetUnits
    constrained_output_model = ConstrainedOutputModel

    @property
    def definition(self):
        definition = """
        -> self.target_fn_table 
        -> self.target_unit_table
        objective_hash:     varchar(64)
        ---
        objective_comment:  varchar(128)
        """
        return definition

    def make(self, key):
        objective_hash = make_hash([key["target_hash"], key["unit_hash"]])
        comments = []
        comments.append((self.target_fn_table & key).fetch1("target_comment"))
        comments.append((self.target_unit_table & key).fetch1("unit_comment"))

        key["objective_comment"] = ", ".join(comments)
        key["objective_hash"] = objective_hash
        self.insert1(key)

    def get_output_selected_model(
        self, model: Module, key: Key
    ) -> constrained_output_model:
        target_fn = (self.target_fn_table & key).get_target_fn()
        unit_ids, data_key = (self.target_unit_table & key).fetch1(
            "unit_ids", "data_key"
        )
        return self.constrained_output_model(
            model, unit_ids, target_fn, forward_kwargs=dict(data_key=data_key)
        )


@schema
class MEITextures(mixins.MEITemplateMixin, dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    trained_model_table = Ensemble
    selector_table = MEIObjective
    method_table = MethodGroup
    seed_table = MEISeed


@schema
class MEIPrototype(mixins.MEITemplateMixin, dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    trained_model_table = SharedReadoutTrainedEnsembleModel
    selector_table = MEIObjective
    method_table = MethodGroup
    seed_table = MEISeed


@schema
class TransferEnsembleModel(mixins.TrainedEnsembleModelTemplateMixin, dj.Manual):
    dataset_table = Dataset
    trained_model_table = TrainedTransferModel

    class Member(mixins.TrainedEnsembleModelTemplateMixin.Member, dj.Part):
        pass


@schema
class MEITransfer(mixins.MEITemplateMixin, dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    trained_model_table = TransferEnsembleModel
    selector_table = Recording.Units
    method_table = Method
    seed_table = MEISeed
