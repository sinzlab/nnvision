import types
import warnings
import numpy as np
import math


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
        limited_output.append(repetitions[:repeat_limit, ] if not randomize else repetitions[
            np.random.choice(n_repeats, repeat_limit if repeat_limit < n_repeats else n_repeats, replace=False)])
    return limited_output


def is_ensemble_function(model):
    return (isinstance(model, types.FunctionType))


def get_cosine_mask(width, height, pixelsPerDegree, fadeStartDegrees):

    fadeStartPixels = round(fadeStartDegrees * pixelsPerDegree)
    widthBegin = -(width / 2)
    widthEnd = (width / 2)
    heightBegin = -(width / 2)
    heightEnd = (width / 2)

    # Make grid of values for entire extent of image
    x, y = np.meshgrid(np.arange(widthBegin, widthEnd), np.arange(heightBegin, heightEnd))
    thresholdRadius = fadeStartPixels / 2
    # Take norm of grid because disk will be circular
    normXY = np.sqrt(np.square(x) + np.square(y))
    remainingRadius = (width - fadeStartPixels) / 2
    disk = np.zeros((width, height))
    for i in range(len(normXY)):
        for j in range(len(normXY)):
            value = int(int(normXY[i, j] >= thresholdRadius) & int(normXY[i, j] <= width / 2))
            disk[i, j] = int(value)
    alteredDisk = disk * (normXY - thresholdRadius) / remainingRadius
    fill = np.zeros((width, height))
    for i in range(len(normXY)):
        for j in range(len(normXY)):
            value = int(normXY[i, j] > width / 2)
            fill[i, j] = int(value)
    finalDisk = (math.pi) / 2 * (alteredDisk + fill)
    blendingMask = np.cos(finalDisk)

    return blendingMask

