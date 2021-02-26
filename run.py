#!/usr/bin/python3

# DJ and utils imports
import datajoint as dj
import time
import tarfile
from shutil import copyfile
import os


dj.config["enable_python_native_blobs"] = True
dj.config["nnfabrik.schema_name"]= 'nnfabrik_toy_V4'
schema = dj.schema('nnfabrik_toy_V4')


# copy data from QB to $SCRATCH volume
os.makedirs('/data/monkey/toliaslab/')
os.mkdir('/data/torch/')

source = '/sinz_shared/monkey/toliaslab/monkey_data.tar.gz'
destination = '/data/monkey/toliaslab/monkey_data.tar.gz'
copyfile(source, destination)

tf = tarfile.open('/data/monkey/toliaslab/monkey_data.tar.gz')
tf.extractall('/data/monkey/toliaslab/')

# project specific imports
from nnvision.tables.from_nnfabrik import TrainedModel
from nnvision.tables.from_mei import MEI, Method

unit_key = dict(unit_type=1, mei_seed=10, ensemble_hash='ce92f09b67d6e154676af577bb9488d5')
method_hashes = (Method & "method_ts > '2021-02-15'" & "method_ts < '2021-02-17'").fetch("method_hash", as_dict=True)
mei_keys = dj.AndList([unit_key,method_hashes])
MEI.populate(mei_keys, display_progress=True, reserve_jobs=True, order="random")
