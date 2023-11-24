"""
this script will run the following analysis:
- load a PyTorch model
- get meis
- for mei:
   - get the corresponding image and augment it:
   - get 15x15 translations in height in width
   - get 30 rotations for each translation
   - show all 15x15x30 images to the model and get all predictions

- for each neuron, get the maximum prediction for each MEI
- order neurons by cluster_id, and show the predictions in a large matrix
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import rotate
from scipy.ndimage import shift
import torch

from nnvision.models.trained_models.v4_data_driven import (
    v4_multihead_attention_ensemble_model,
)

model = v4_multihead_attention_ensemble_model
model.eval().cuda()

pop_key = {
    "model_fn": "nnvision.models.simclr.ConvNextFC",
    "model_hash": "6510835da772e26b8ceb7013718f367f",
    "dataset_fn": "nnvision.datasets.mei_loaders.mei_tsmincne_loader",
    "dataset_hash": "5ea066fad56931cbfca1fceb5ac3e5ca",
    "trainer_fn": "nnvision.training.simclr.tsimcne_trainer",
    "trainer_hash": "83f17636d44655570809589946dd296d",
    "seed": 1000,
}

dl = (Dataset & pop_key).get_dataloader()
unit_ids = dl[
    "validation"
].dataset.units  # array of shape (n_units*n_seeds,), 11000units, with 10-20 seeds each
unique_unit_ids = np.unique(unit_ids)  # array of shape (n_units,), 11000 units

# get the MEI for each unit
meis = dl[
    "validation"
].dataset.meis  # array of shape (n_units*n_seeds,1,100,100), 11000units, with 10-20 seeds each, 1 channel, 100x100 pixels

# create a for loop:
# for each mei, get the 15x15 translations in height and width, and the translations

response_array = []
for img in meis:
    # get the 15x15 translations in height and width, and the translations
    # get 15x15 translations in height in width
    # get 30 rotations for each translation
    # show all 15x15x30 images to the model and get all predictions
    # for each neuron, get the maximum prediction for each MEI
    # order neurons by cluster_id, and show the predictions in a large matrix

    # get the 15x15 translations in height in width
    # get 30 rotations for each translation
    # show all 15x15x30 images to the model and get all predictions

    augmented_imgs = []
    for i in range(15):
        for j in range(15):
            for k in range(30):
                img_rot = rotate(img.numpy(), k * 6, reshape=False)
                img_shift = shift(img_rot, [i * 2, j * 2, 0])
                augmented_imgs.append(img_shift)

    augmented_imgs = torch.from_numpy(np.array(augmented_imgs))
    with torch.no_grad():
        outputs = model(augmented_imgs.cuda(), data_key="all_sessions")
        outputs = outputs.cpu().numpy().max(0)
    response_array.append(outputs)
