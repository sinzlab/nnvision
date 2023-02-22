import numpy as np
import torch

from cnexp.losses.infonce import InfoNCECauchy
from cnexp.lrschedule import CosineAnnealingSchedule, LinearAnnealingSchedule
from cnexp.models.mutate_model import mutate_model
from cnexp.optimizers import make_sgd
from cnexp.train import train


def tsimcne_trainer(
    model,
    dataloaders,
    seed,
    n_epochs1=1000,
    n_epochs2=50,
    n_epochs3=500,
    optimizer="sgd",
    lr1=1e-3,
    lr2=1e-3,
    lr3=1e-5,
    scheduler="cosine",
    disable_tqdm=False,
    store_z_n_epochs=10,
    cb=None,
    sgd_step3_lr_factor=1,
    runs_per_epoch=1,
    dei_val_loss=None,
    **kwargs,
):
    rng = np.random.default_rng(1)
    # train loader for mei or for cifar
    train_loader = (
        dataloaders["train_contrastive"]
        if "train_contrastive" in dataloaders
        else dataloaders["train_contrastive_loader"]
    )
    val_loader = (
        dataloaders["validation"]
        if "validation" in dataloaders
        else dataloaders["full_plain_loader"]
    )
    dei_loader = (
        dataloaders["dei_batch_loader"]
        if "dei_batch_loader" in dataloaders
        else None
    )

    loss = InfoNCECauchy()
    # set up return values

    losses, features, distances = [], [], []

    ## Step 1
    # get optimizer and scheduler
    if optimizer == "sgd":
        opt = make_sgd(model, batch_size=train_loader.batch_size)
    elif optimizer == "adamW":
        opt = torch.optim.AdamW(model.parameters(), lr=lr1)

    if scheduler == "cosine":
        lrsched = CosineAnnealingSchedule(opt=opt, n_epochs=n_epochs1)
    elif scheduler == "linear":
        lrsched = LinearAnnealingSchedule(opt, n_epochs=n_epochs1)
    else:
        raise ValueError(
            f"scheduler {scheduler} not implemented. Choose from ['cosine', 'linear]"
        )

    out = train(
        train_loader,
        model,
        criterion=loss,
        opt=opt,
        lrsched=lrsched,
        device="cuda",
        seed=rng.integers(seed),
        checkpoint="checkpoint",
        checkpoint_valid=False,
        disable_tqdm=disable_tqdm,
        call_back=cb,
        runs_per_epoch=runs_per_epoch,
        dei_val_loss=dei_val_loss,
        dei_loader=dei_loader,
    )
    losses.append(np.hstack(out["losses"]))
    features.append(out["zs"])
    distances.append(np.hstack(out["distances"]))

    ## Step 2
    model = mutate_model(model, change="lastlin", freeze=True, out_dim=2)
    # get optimizer and scheduler
    if optimizer == "sgd":
        opt = make_sgd(model, batch_size=train_loader.batch_size)
    elif optimizer == "adamW":
        opt = torch.optim.AdamW(model.parameters(), lr=lr2)

    if scheduler == "cosine":
        lrsched = CosineAnnealingSchedule(opt=opt, n_epochs=n_epochs2)
    elif scheduler == "linear":
        lrsched = LinearAnnealingSchedule(opt, n_epochs=n_epochs2)
    else:
        raise ValueError(
            f"scheduler {scheduler} not implemented. Choose from ['cosine', 'linear]"
        )

    out = train(
        train_loader,
        model,
        criterion=loss,
        opt=opt,
        lrsched=lrsched,
        device="cuda",
        seed=rng.integers(seed),
        checkpoint="checkpoint",
        checkpoint_valid=False,
        disable_tqdm=disable_tqdm,
        call_back=cb,
        runs_per_epoch=runs_per_epoch,
        dei_val_loss=dei_val_loss,
        dei_loader=dei_loader,
    )
    losses.append(np.hstack(out["losses"]))
    features.append(out["zs"])
    distances.append(np.hstack(out["distances"]))

    ## Step 3
    model.backbone.requires_grad_(True)
    model.projection_head.requires_grad_(True)

    # get optimizer and scheduler
    if optimizer == "sgd":
        if sgd_step3_lr_factor is not None:
            initial_lr = opt.param_groups[0]["lr"]
            opt = make_sgd(model, batch_size=train_loader.batch_size, lr=initial_lr / sgd_step3_lr_factor)
        else:
            opt = make_sgd(model, batch_size=train_loader.batch_size)
    elif optimizer == "adamW":
        opt = torch.optim.AdamW(model.parameters(), lr=lr3)

    if scheduler == "cosine":
        lrsched = CosineAnnealingSchedule(opt=opt, n_epochs=n_epochs3)
    elif scheduler == "linear":
        lrsched = LinearAnnealingSchedule(opt, n_epochs=n_epochs3)
    else:
        raise ValueError(
            f"scheduler {scheduler} not implemented. Choose from ['cosine', 'linear]"
        )

    out = train(
        train_loader,
        model,
        criterion=loss,
        opt=opt,
        lrsched=lrsched,
        device="cuda",
        seed=rng.integers(seed),
        checkpoint="checkpoint",
        checkpoint_valid=False,
        store_Z_n_epochs=store_z_n_epochs,
        plain_dataloader=val_loader,
        disable_tqdm=disable_tqdm,
        call_back=cb,
        runs_per_epoch=runs_per_epoch,
        dei_val_loss=dei_val_loss,
        dei_loader=dei_loader,
    )
    losses.append(np.hstack(out["losses"]))
    features.append(out["zs"])
    distances.append(np.hstack(out["distances"]))

    output = {"losses": losses, "features": features, "distances": distances}
    score = np.array(out["losses"][-1], dtype="single").item()
    return score, output, model.state_dict()
