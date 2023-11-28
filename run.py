#!/usr/bin/python

import numpy as np
from time import sleep
sleep(np.random.rand()*5)

import os
import datajoint as dj
dj.config["database.host"] = '134.76.19.44'
os.environ["MINIO_ACCESS_KEY"] = "OOZ67ZZ3IC594HVU39MM"
os.environ["MINIO_SECRET_KEY"] = "IbjiNZdvODrzgzM4J3wBcutgCnBkI2CGt4KFpHxL"

dj.config["enable_python_native_blobs"] = True
dj.config['nnfabrik.schema_name'] = "nnfabrik_toy_V4"

from nnfabrik.main import *
from nnvision.tables.from_nnfabrik import TrainedModel


dataset_keys = [dict(dataset_hash="9ef1991a6c99e7d5af6e2a51c3a537a6"),
                dict(dataset_hash="bc288bef8512a5e1b87acb5c6f4de99a"),
                dict(dataset_hash="7214ff189a2fb06af589d9a96e2cf140"),
               ]
trainer_keys = dict(trainer_hash="e145fd0f8537a754b8972d894ed2820d")
model_keys = (Model & "model_ts > '2023-11-28 05'").fetch("KEY")
pop_key = dj.AndList([dataset_keys, trainer_keys, model_keys, dict(seed=1000)])

TrainedModel().populate(pop_key,
                        display_progress=True,
                        reserve_jobs=True,
                        order="random",)