import warnings
import numpy as np
import torch
from mlutils.measures import corr
from mlutils.training import eval_state, device_state
import types
import contextlib


def model_predictions_repeats(model, dataloader, data_key, device='cpu', broadcast_to_target=False):
    """
    Computes model predictions for dataloader that yields batches with identical inputs along the first dimension
    Unique inputs will be forwarded only once through the model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons as a list: [num_images][num_reaps, num_neurons]
        output: responses as predicted by the network for the unique images. If broadcast_to_target, returns repeated 
                outputs of shape [num_images][num_reaps, num_neurons] else (default) returns unique outputs of shape [num_images, num_neurons]
    """
    
    output = torch.empty(0)
    target = []
    
    # Get unique images and concatenate targets:
    unique_images = torch.empty(0)
    for images, responses in dataloader:
        if len(images.shape) == 5:
            images = images.squeeze(dim=0)
            responses = responses.squeeze(dim=0)
        
        assert torch.all(torch.eq(images[-1,], images[0,],)), "All images in the batch should be equal"
        unique_images = torch.cat((unique_images, images[0:1, ]), dim=0)
        target.append(responses.detach().cpu().numpy())
    
    # Forward unique images once:
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        with device_state(model, device) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
            output = model(unique_images.to(device), data_key=data_key).detach().cpu()
    
    output = output.numpy()   
        
    if broadcast_to_target:
        output = [np.broadcast_to(x, target[idx].shape) for idx, x in enumerate(output)]
    
    
    return target, output
    

def model_predictions(model, dataloader, data_key, device='cpu'):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network
    """

    target, output = torch.empty(0), torch.empty(0)
    for images, responses in dataloader:
        if len(images.shape) == 5:
            images = images.squeeze(dim=0)
            responses = responses.squeeze(dim=0)
        with torch.no_grad():
            with device_state(model, device) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
                output = torch.cat((output, (model(images.to(device), data_key=data_key).detach().cpu())), dim=0)
            target = torch.cat((target, responses.detach().cpu()), dim=0)

    return target.numpy(), output.numpy()



def get_avg_correlations(model, dataloaders, device='cpu', as_dict=False, per_neuron=True, **kwargs):
    """
    Returns correlation between model outputs and average responses over repeated trials
    
    """
    if 'test' in dataloaders:
        dataloaders = dataloaders['test']
    
    correlations = {}
    for k, loader in dataloaders.items():

        # Compute correlation with average targets
        target, output = model_predictions_repeats(dataloader=loader, model=model, data_key=k, device=device, broadcast_to_target=False)
        target_mean = np.array([t.mean(axis=0) for t in target])
        correlations[k] = corr(target_mean, output, axis=0)
        
        # Check for nans
        if np.any(np.isnan(correlations[k])):
            warnings.warn('{}% NaNs , NaNs will be set to Zero.'.format(np.isnan(correlations[k]).mean() * 100))
        correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = np.hstack([v for v in correlations.values()]) if per_neuron else np.mean(np.hstack([v for v in correlations.values()]))
    return correlations


def get_correlations(model, dataloaders, device='cpu', as_dict=False, per_neuron=True, **kwargs):
    correlations = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            target, output = model_predictions(dataloader=v, model=model, data_key=k, device=device)
            correlations[k] = corr(target, output, axis=0)

            if np.any(np.isnan(correlations[k])):
                warnings.warn('{}% NaNs , NaNs will be set to Zero.'.format(np.isnan(correlations[k]).mean() * 100))
            correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = np.hstack([v for v in correlations.values()]) if per_neuron else np.mean(np.hstack([v for v in correlations.values()]))
    return correlations


def get_poisson_loss(model, dataloaders, device='cpu', as_dict=False, avg=False, per_neuron=True, eps=1e-12):
    poisson_loss = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            target, output = model_predictions(dataloader=v, model=model, data_key=k, device=device)
            loss = output - target * np.log(output + eps)
            poisson_loss[k] = np.mean(loss, axis=0) if avg else np.sum(loss, axis=0)
    if as_dict:
        return poisson_loss
    else:
        if per_neuron:
            return np.hstack([v for v in poisson_loss.values()])
        else:
            return np.mean(np.hstack([v for v in poisson_loss.values()])) if avg else np.sum(np.hstack([v for v in poisson_loss.values()]))


def get_repeats(dataloader, min_repeats=2):
    # save the responses of all neuron to the repeats of an image as an element in a list
    repeated_inputs = []
    repeated_outputs = []
    for inputs, outputs in dataloader:
        if len(inputs.shape) == 5:
            inputs = np.squeeze(inputs.cpu().numpy(), axis=0)
            outputs = np.squeeze(outputs.cpu().numpy(), axis=0)
        else:
            inputs = inputs.cpu().numpy()
            outputs = outputs.cpu().numpy()
        r, n = outputs.shape  # number of frame repeats, number of neurons
        if r < min_repeats:  # minimum number of frame repeats to be considered for oracle, free choice
            continue
        assert np.all(np.abs(np.diff(inputs, axis=0)) == 0), "Images of oracle trials do not match"
        repeated_inputs.append(inputs)
        repeated_outputs.append(outputs)
    return np.array(repeated_inputs), np.array(repeated_outputs)


def get_oracles(dataloaders, as_dict=False, per_neuron=True):
    oracles = {}
    for k, v in dataloaders.items():
        _, outputs = get_repeats(v)
        oracles[k] = compute_oracle_corr(np.array(outputs))
    if not as_dict:
        oracles = np.hstack([v for v in oracles.values()]) if per_neuron else np.mean(np.hstack([v for v in oracles.values()]))
    return oracles


def compute_oracle_corr(repeated_outputs):
    if len(repeated_outputs.shape) == 3:
        _, r, n = repeated_outputs.shape
        oracles = (repeated_outputs.mean(axis=1, keepdims=True) - repeated_outputs / r) * r / (r - 1)
        if np.any(np.isnan(oracles)):
            warnings.warn('{}% NaNs when calculating the oracle. NaNs will be set to Zero.'.format(np.isnan(oracles).mean() * 100))
        oracles[np.isnan(oracles)] = 0
        return corr(oracles.reshape(-1, n), repeated_outputs.reshape(-1, n), axis=0)
    else:
        oracles = []
        for outputs in repeated_outputs:
            r, n = outputs.shape
            # compute the mean over repeats, for each neuron
            mu = outputs.mean(axis=0, keepdims=True)
            # compute oracle predictor
            oracle = (mu - outputs / r) * r / (r - 1)

            if np.any(np.isnan(oracle)):
                warnings.warn('{}% NaNs when calculating the oracle. NaNs will be set to Zero.'.format(
                    np.isnan(oracle).mean() * 100))
                oracle[np.isnan(oracle)] = 0

            oracles.append(oracle)
        return corr(np.vstack(repeated_outputs), np.vstack(oracles), axis=0)


def get_explainable_var(dataloaders, as_dict=False, per_neuron=True):
    explainable_var = {}
    for k ,v in dataloaders.items():
        _, outputs = get_repeats(v)
        explainable_var[k] = compute_explainable_var(outputs)
    if not as_dict:
        explainable_var = np.hstack([v for v in explainable_var.values()]) if per_neuron else np.mean(np.hstack([v for v in explainable_var.values()]))
    return explainable_var


def compute_explainable_var(outputs, eps=1e-9):
    ImgVariance = []
    TotalVar = np.var(np.vstack(outputs), axis=0, ddof=1)
    for out in outputs:
        ImgVariance.append(np.var(out, axis=0, ddof=1))
    ImgVariance = np.vstack(ImgVariance)
    NoiseVar = np.mean(ImgVariance, axis=0)
    explainable_var = (TotalVar - NoiseVar) / (TotalVar + eps)
    return explainable_var


def get_FEV(model, dataloaders, device='cpu', as_dict=False, per_neuron=True):
    FEV = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            repeated_inputs, repeated_outputs = get_repeats(v)
            FEV[k] = compute_FEV(repeated_inputs=repeated_inputs,
                                 repeated_outputs=repeated_outputs,
                                 model=model,
                                 device=device,
                                 data_key=k,
                                 )
    if not as_dict:
        FEV = np.hstack([v for v in FEV.values()]) if per_neuron else np.mean(np.hstack([v for v in FEV.values()]))
    return FEV


def compute_FEV(repeated_inputs, repeated_outputs, model, data_key=None, device='cpu', return_exp_var=False):

    ImgVariance = []
    PredVariance = []
    for i, outputs in enumerate(repeated_outputs):
        inputs = repeated_inputs[i]
        predictions = model(torch.tensor(inputs).to(device), data_key=data_key).detach().cpu().numpy()
        PredVariance.append((outputs - predictions) ** 2)
        ImgVariance.append(np.var(outputs, axis=0, ddof=1))

    PredVariance = np.vstack(PredVariance)
    ImgVariance = np.vstack(ImgVariance)

    TotalVar = np.var(np.vstack(repeated_outputs), axis=0, ddof=1)
    NoiseVar = np.mean(ImgVariance, axis=0)
    FEV = (TotalVar - NoiseVar) / TotalVar

    PredVar = np.mean(PredVariance, axis=0)
    FEVe = 1 - (PredVar - NoiseVar) / (TotalVar - NoiseVar)
    return [FEV, FEVe] if return_exp_var else FEVe


def get_cross_oracles(data, reference_data):
    _, outputs = get_repeats(data)
    _, outputs_reference = get_repeats(reference_data)
    cross_oracles = compute_cross_oracles(outputs, outputs_reference)
    return cross_oracles


def compute_cross_oracles(repeats, reference_data):
    pass


def normalize_RGB_channelwise(mei):
    mei_copy = mei.copy()
    mei_copy = mei_copy - mei_copy.min(axis=(1, 2), keepdims=True)
    mei_copy = mei_copy / mei_copy.max(axis=(1, 2), keepdims=True)
    return mei_copy


def normalize_RGB(mei):
    mei_copy = mei.copy()
    mei_copy = mei_copy - mei_copy.min()
    mei_copy = mei_copy / mei_copy.max()
    return mei_copy
