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
source = '/sinz_shared/monkey/toliaslab/monkey_data.tar.gz'
destination = '/data/monkey/toliaslab/monkey_data.tar.gz'

start = time.time()
copyfile(source, destination)
end = time.time()
print("time spent copying: ", end - start)

start = time.time()
tf = tarfile.open('/data/monkey/toliaslab/monkey_data.tar.gz')
tf.extractall()
end = time.time()
print("time spent unzipping monkey data: ", end - start)

# verify extraction
print("... content of monkey data ... ")
print(os.listdir('/data/monkey/toliaslab/'))


# project specific imports
from nnvision.tables.from_nnfabrik import TrainedModel
print("Entries in TrainedModel table", len(TrainedModel()))