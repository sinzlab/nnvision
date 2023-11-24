#!/usr/bin/python

import numpy as np
from time import sleep
sleep(np.random.rand()*10)

import os
import datajoint as dj
dj.config["database.host"] = '134.76.19.44'
os.environ["MINIO_ACCESS_KEY"] = "OOZ67ZZ3IC594HVU39MM"
os.environ["MINIO_SECRET_KEY"] = "IbjiNZdvODrzgzM4J3wBcutgCnBkI2CGt4KFpHxL"

dj.config["enable_python_native_blobs"] = True
dj.config['nnfabrik.schema_name'] = "nnfabrik_toy_V4"
schema = dj.schema("nnfabrik_toy_V4")


from nnvision.tables.from_mei import MEI, MEISeed
from nnvision.tables.ensemble_scores import CorrelationToAverageEnsembleScore

# 50 seeds
mei_seeds = MEISeed().fetch("KEY", order_by="mei_seed", limit=50)
# 1100 units
unit_key = (CorrelationToAverageEnsembleScore.Units & dict(ensemble_hash="b2fb11d8f9206374f0fcb9bf6f1569cb")).fetch("KEY", order_by="unit_avg_correlation DESC", limit=1100)

mei_key = dj.AndList([unit_key,
                      mei_seeds,
                      dict(method_hash="3bb4e453e0f368d436803d8bac68b22d")
                      ])

MEI().populate(mei_key, display_progress=True, reserve_jobs=True, order="random",)
