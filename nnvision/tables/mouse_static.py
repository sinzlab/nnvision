import datajoint as dj
import os
import numpy as np
from torch.utils.data import DataLoader
from mlutils.data.datasets import StaticImageSet
from mlutils.data.samplers import RepeatsBatchSampler
from mlutils.measures import corr
from tqdm import tqdm
schema = dj.schema("sinzlab_data_static", locals())


@schema
class StaticScan(dj.Computed):
    definition = """
    # gatekeeper for scan and preprocessing settings
    animal_id            : int                          # id number
    session              : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    pipe_version         : smallint                     #
    segmentation_method  : tinyint                      #
    spike_method         : tinyint                      # spike inference method    
    """

    class Unit(dj.Part):
        definition = """
        # smaller primary key table for data
        -> master        
        unit_id              : int                          # unique per scan & segmentation method
        ---
        """

    class UnitClusters(dj.Part):
        definition = """
        # Unit clusters which contain possibly identical units with correlation threshold 0.7 
        -> master 
        -> master.Unit
        cluster_id60           : int               # threshold: 0.6 correlation 
        cluster_id70           : int               # threshold: 0.7 correlation
        ---
        """


@schema
class Preprocessing(dj.Lookup):
    definition = """
    # settings for movie preprocessing
    preproc_id       : tinyint # preprocessing ID
    ---
    offset           : decimal(6,4)  # offset to stimulus onset in s
    duration         : decimal(6,4)  # window length in s
    row              : smallint       # row size of movies
    col              : smallint       # col size of movie
    filter           : varchar(24)   # filter type for window extraction
    """


@schema
class HDF5File(dj.Manual):
    definition = """
    # HDF5-File
    -> StaticScan    
    -> Preprocessing
    ---
    path                 : attach@minio_static                # path in database    
    comment=NULL         : varchar(1000)                      # Optional comment for this entry
    """


@schema
class Oracle(dj.Computed):
    definition = """
    # oracle computation for static images
    -> HDF5File
    ---
    pearson       : float         # mean test correlation averaged over units
    """

    class UnitScores(dj.Part):
        definition = """
        -> master
        -> StaticScan.Unit
        ---
        unit_pearson           : float     # mean test correlation per unit
        """

    class GroundTruthUnitScores(dj.Part):
        definition = """
        # This is only for simulated data where we have ground truths
        -> master
        -> StaticScan.Unit
        ---
        unit_pearson           : float     # mean test correlation per unit, based on ground truth
        """

    def make(self, key):
        # --- load data
        path = (HDF5File & key).fetch1("path")
        dataset = StaticImageSet(path, "images", "responses")
        os.remove(path)
        types = np.unique(dataset.types)
        if len(types) == 1 and types[0] == "stimulus.Frame":
            condition_hashes = dataset.info.frame_image_id
        else:
            raise ValueError("Do not recognize types={}".format(*types))
        loader = DataLoader(dataset, sampler=RepeatsBatchSampler(condition_hashes))
        # --- compute oracles
        oracles, data = [], []
        for inputs, outputs in loader:
            inputs = np.squeeze(inputs.cpu().numpy(), axis=0)
            outputs = np.squeeze(outputs.cpu().numpy(), axis=0)
            r, n = outputs.shape  # number of frame repeats, number of neurons
            if r < 4:  # minimum number of frame repeats to be considered for oracle, free choice
                continue
            assert np.all(np.abs(np.diff(inputs, axis=0)) == 0), "Images of oracle trials do not match"
            mu = outputs.mean(axis=0, keepdims=True)
            oracle = (mu - outputs / r) * r / (r - 1)
            oracles.append(oracle)
            data.append(outputs)
        assert len(data) > 0, "Found no oracle trials!"
        pearson = corr(np.vstack(data), np.vstack(oracles), axis=0)
        unit_ids = dataset.neurons.unit_ids
        assert len(unit_ids) == len(pearson) == outputs.shape[-1], "Neuron numbers do not add up"
        self.insert1(dict(key, pearson=np.mean(pearson)), ignore_extra_fields=True)
        self.UnitScores().insert(
            [dict(key, unit_pearson=c, unit_id=u) for u, c in tqdm(zip(unit_ids, pearson), total=len(unit_ids))],
            ignore_extra_fields=True,
        )
        # This is for simulated data
        if key["animal_id"] == 0:
            test_idx = dataset.tiers == "test"
            gt_pearson = corr(
                dataset.responses[test_idx], dataset.simulation_info["ground_truths"][:][test_idx], axis=0
            )
            self.GroundTruthUnitScores().insert(
                [dict(key, unit_pearson=c, unit_id=u) for u, c in tqdm(zip(unit_ids, gt_pearson), total=len(unit_ids))],
                ignore_extra_fields=True,
            )