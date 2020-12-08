import datajoint as dj

dj.config["enable_python_native_blobs"] = True
dj.config["nnfabrik.schema_name"]= 'nnfabrik_schema'
schema = dj.schema('nnfabrik_schema')
from nnvision.tables.from_nnfabrik import TrainedModel

keys = {}
TrainedModel.populate(keys, display_progress=True, reserve_jobs=True)
