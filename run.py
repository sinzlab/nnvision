#!/usr/bin/python3

import datajoint as dj

dj.config["enable_python_native_blobs"] = True
dj.config["nnfabrik.schema_name"]= 'nnfabrik_toy_V4'
schema = dj.schema('nnfabrik_toy_V4')
from nnvision.tables.from_nnfabrik import TrainedModel

keys = {}

import os

os.listdir('/sinz_shared')

print("Entries in TrainedModel table", len(TrainedModel()))