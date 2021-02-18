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

pop_key = dj.AndList([{'dataset_hash': '8740d18cb1951608c573ceda09d47aef',
            'trainer_fn': 'nnvision.training.trainers.nnvision_trainer',
            'trainer_hash': '7eba3d5e8d426d6bbcd3f248565f8cfb'},
           [{'model_hash': 'ade1c26ff74aef5479499079a219474e'},
            {'model_hash': 'ea5ee15d1e4431417f3f01f5d4bca191'}]])

TrainedModel().populate(pop_key, display_progress=True, reserve_jobs=True)