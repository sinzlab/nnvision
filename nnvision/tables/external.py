import os

import datajoint as dj

bcm_conn = dj.Connection(os.environ['BCM_HOST'], os.environ['BCM_USER'], os.environ['BCM_PASS'])
stim = dj.create_virtual_module("stimulus", "pipeline_stimulus", connection=bcm_conn)
