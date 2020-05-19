import warnings
import numpy as np


def get_subset_of_repeats(outputs, repeat_limit, randomize=True):
    """
    Args:
        outputs (array or list): repeated responses/targets to the same input. with the shape [inputs, ] [reps, neurons]
                                    or array(inputs, reps, neurons)
        repeat_limit (int): how many reps are selected
        randomize (cool): if True, takes a random selection of repetitions. if false, takes the first n repetitions.

    Returns: limited_outputs (list): same shape as inputs, but with reduced number of repetitions

    """
    limited_output = []
    for repetitions in outputs:
        n_repeats = repetitions.shape[0]
        if repeat_limit > n_repeats:
            warnings.warn(f"Repeat limit of {repeat_limit}is larger than the repeats of {n_repeats}")
        limited_output.append(repetitions[:repeat_limit, ] if not randomize else repetitions[
            np.random.choice(n_repeats, repeat_limit, replace=False)])
    return limited_output