import warnings
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from collections import Iterable

from neuralpredictors.measures import *
from neuralpredictors import measures as mlmeasures
from neuralpredictors.training import (
    early_stopping,
    MultipleObjectiveTracker,
    eval_state,
    cycle_datasets,
    Exhauster,
    LongCycler,
)
from nnfabrik.utility.nn_helpers import set_random_seed

from ..utility import measures
from ..utility.measures import get_correlations, get_poisson_loss


def nnvision_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,  # trainer args
    loss_function="PoissonLoss",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,  # early stopping args
    max_iter=100,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,  # lr scheduler args
    cb=None,
    track_training=False,
    return_test_score=False,
    batchping=1000,
    **kwargs,
):
    """

    Args:
        model:
        dataloaders:
        seed:
        avg_loss:
        scale_loss:
        loss_function:
        stop_function:
        loss_accum_batch_n:
        device:
        verbose:
        interval:
        patience:
        epoch:
        lr_init:
        max_iter:
        maximize:
        tolerance:
        restore_best:
        lr_decay_steps:
        lr_decay_factor:
        min_lr:
        cb:
        track_training:
        **kwargs:

    Returns:

    """

    def full_objective(model, dataloader, data_key, *args, **kwargs):
        """

        Args:
            model:
            dataloader:
            data_key:
            *args:

        Returns:

        """
        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        preds = model(args[0].to(device), data_key=data_key, **kwargs)
        if "bools" in kwargs: #zero out predictions where bools are False, a.k.a where neurons weren't recorded
            preds = preds * kwargs["bools"].to(device)
        resps = args[1].to(device)
        return loss_scale * criterion(preds, resps) + model.regularizer(data_key)

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(mlmeasures, loss_function)(avg=avg_loss)
    stop_closure = partial(
        getattr(measures, stop_function),
        dataloaders=dataloaders["validation"],
        device=device,
        per_neuron=False,
        avg=True,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )

    if track_training:
        tracker_dict = dict(
            correlation=partial(
                get_correlations,
                model=model,
                dataloaders=dataloaders["validation"],
                device=device,
                per_neuron=False,
            ),
            poisson_loss=partial(
                get_poisson_loss,
                model=model,
                dataloaders=dataloaders["validation"],
                device=device,
                per_neuron=False,
                avg=False,
            ),
        )
        if hasattr(model, "tracked_values"):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

    # train over epochs
    for epoch, val_obj in early_stopping(
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=epoch,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        tracker=tracker,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):

        # print the quantities from tracker
        if verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad()
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            loss = full_objective(
                model, dataloaders["train"], data_key, *data[:2], **data._asdict()
            )
            loss.backward()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (batch_no % batchping == 0) and (cb is not None):
                cb()

    ##### Model evaluation ####################################################################################################
    model.eval()
    tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False
    )
    if return_test_score:
        test_correlation = get_correlations(
            model, dataloaders["test"], device=device, as_dict=False, per_neuron=False
        )

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()} if track_training else {}
    output["validation_corr"] = validation_correlation

    score = (
        np.mean(test_correlation)
        if return_test_score
        else np.mean(validation_correlation)
    )
    return score, output, model.state_dict()


def finetune_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,  # trainer args
    loss_function="PoissonLoss",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,  # early stopping args
    max_iter=100,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,  # lr scheduler args
    cb=None,
    track_training=False,
    return_test_score=False,
    fine_tune="sequential",
    **kwargs,
):
    def full_objective(model, dataloader, data_key, *args):

        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        return loss_scale * criterion(
            model(args[0].to(device), data_key=data_key), args[1].to(device)
        ) + model.regularizer(data_key)

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(mlmeasures, loss_function)(avg=avg_loss)
    stop_closure = partial(
        getattr(measures, stop_function),
        dataloaders=dataloaders["validation"],
        device=device,
        per_neuron=False,
        avg=True,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )

    if track_training:
        tracker_dict = dict(
            correlation=partial(
                get_correlations,
                model=model,
                dataloaders=dataloaders["validation"],
                device=device,
                per_neuron=False,
            ),
            poisson_loss=partial(
                get_poisson_loss,
                model=model,
                dataloaders=dataloaders["validation"],
                device=device,
                per_neuron=False,
                avg=False,
            ),
        )
        if hasattr(model, "tracked_values"):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

    if fine_tune == "sequential":
        parameters_to_train = [model.readout.parameters(), model.parameters()]
    elif fine_tune == "full":
        parameters_to_train = [model.parameters()]
    elif fine_tune == "core":
        parameters_to_train = [model.core.parameters()]
    elif fine_tune == "readout":
        parameters_to_train = [model.readout.parameters()]

    for i, parameters in enumerate(parameters_to_train):
        if isinstance(lr_init, Iterable):
            lr = lr_init[i]
        print(f"training with lr = {lr}")
        optimizer = torch.optim.Adam(parameters, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if maximize else "min",
            factor=lr_decay_factor,
            patience=patience,
            threshold=tolerance,
            min_lr=min_lr,
            verbose=verbose,
            threshold_mode="abs",
        )
        # train over epochs
        for epoch, val_obj in early_stopping(
            model,
            stop_closure,
            interval=interval,
            patience=patience,
            start=0,
            max_iter=max_iter,
            maximize=maximize,
            tolerance=tolerance,
            restore_best=restore_best,
            tracker=tracker,
            scheduler=scheduler,
            lr_decay_steps=lr_decay_steps,
        ):

            # print the quantities from tracker
            if verbose and tracker is not None:
                print("=======================================")
                for key in tracker.log.keys():
                    print(key, tracker.log[key][-1], flush=True)

            # executes callback function if passed in keyword args
            if cb is not None:
                cb()

            # train over batches
            optimizer.zero_grad()
            for batch_no, (data_key, data) in tqdm(
                enumerate(LongCycler(dataloaders["train"])),
                total=n_iterations,
                desc="Epoch {}".format(epoch),
            ):

                loss = full_objective(model, dataloaders["train"], data_key, *data)
                loss.backward()
                if (batch_no + 1) % optim_step_count == 0:
                    optimizer.step()
                    optimizer.zero_grad()

    ##### Model evaluation ####################################################################################################
    model.eval()
    tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False
    )
    test_correlation = get_correlations(
        model, dataloaders["test"], device=device, as_dict=False, per_neuron=False
    )

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()} if track_training else {}
    output["validation_corr"] = validation_correlation

    score = (
        np.mean(test_correlation)
        if return_test_score
        else np.mean(validation_correlation)
    )
    return score, output, model.state_dict()


def shared_readout_trainer(model, dataloaders, seed, uid=None, cb=None):
    score = 0
    output = 0
    model_state = model.state_dict()

    return score, output, model_state
