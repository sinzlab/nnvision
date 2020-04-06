import datajoint as dj
from featurevis.main import TrainedEnsembleModelTemplate, CSRFV1SelectorTemplate, MEIMethod, MEITemplate
from nnfabrik.main import Dataset
from .from_nnfabrik import TrainedModel
from mlutils.data.datasets import StaticImageSet
from featurevis import integration
from ..mei.helpers import get_neuron_mappings, get_real_mappings

schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))

class MouseSelectorTemplate(dj.Computed):
    """CSRF V1 selector table template.

    To create a functional "CSRFV1Selector" table, create a new class that inherits from this template and decorate it
    with your preferred Datajoint schema. By default, the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset

    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    neuron_id       : smallint unsigned # unique neuron identifier
    ---
    neuron_position : smallint unsigned # integer position of the neuron in the model's output 
    session_id      : varchar(13)       # unique session identifier
    """

    _key_source = Dataset & dict(dataset_fn="mouse_static_loaders")

    def make(self, key):
        dataset_config = (Dataset & key).fetch1("dataset_config")

        path = dataset_config["paths"][0]
        dat = StaticImageSet(path, 'images', 'responses')
        neuron_ids = dat.neurons.unit_ids

        data_key = path.split('static')[-1].split('.')[0].replace('preproc', '')

        mappings = []
        for neuron_pos, neuron_id in enumerate(neuron_ids):
            mappings.append(dict(key, neuron_id=neuron_id, neuron_position=neuron_pos, session_id=data_key))

        self.insert(mappings)

    def get_output_selected_model(self, model, key):
        neuron_pos, session_id = (self & key).fetch1("neuron_position", "session_id")
        return integration.get_output_selected_model(neuron_pos, session_id, model)


class MonkeySelectorTemplate(dj.Computed):
    """CSRF V1 selector table template.

    To create a functional "CSRFV1Selector" table, create a new class that inherits from this template and decorate it
    with your preferred Datajoint schema. By default, the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset

    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    neuron_id       : int # unique neuron identifier
    session_id      : varchar(13)       # unique session identifier
    ---
    neuron_position : int # integer position of the neuron in the model's output 
    
    """

    _key_source = Dataset & dict(dataset_fn="nnvision.datasets.monkey_static_loader")

    def make(self, key):
        dataset_config = (Dataset & key).fetch1("dataset_config")
        mappings = get_neuron_mappings(dataset_config, key)
        self.insert(mappings)

    def get_output_selected_model(self, model, key):
        neuron_pos, session_id = (self & key).fetch1("neuron_position", "session_id")
        return integration.get_output_selected_model(neuron_pos, session_id, model)


@schema
class TrainedEnsembleModel(TrainedEnsembleModelTemplate):
    dataset_table = Dataset
    trained_model_table = TrainedModel


@schema
class MouseSelector(MouseSelectorTemplate):
    dataset_table = Dataset


@schema
class MonkeySelector(MonkeySelectorTemplate):
    dataset_table = Dataset


@schema
class MEI(MEITemplate):
    trained_model_table = TrainedEnsembleModel
    selector_table = MouseSelector

