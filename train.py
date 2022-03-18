import torch

import torchvision
import numpy as np

from dircn import DatasetContainer
from dircn import DatasetLoader
from dircn import DatasetInfo

from dircn.models import MultiLoss, MultiMetric
from dircn.metrics import (
    NMSE,
    PSNR,
    )

from dircn.models.dircn.dircn import DIRCN


from dircn.trainer import Trainer
from dircn.masks import create_mask_for_mask_type
from dircn.config import ConfigReader

from dircn.models.losses import SSIM, FSSIMLoss

from dircn.preprocessing import (
    ApplyMaskColumn,
    KspaceToImage,
    ComplexNumpyToTensor,
    ComplexAbsolute,
    RSS,
    DownsampleFOV,
    NormalizeKspace,
    )

# Bluemaster
train = DatasetContainer()
train.fastMRI(path='path_to_fastMRI_training', datasetname='fastMRI', dataset_type='training')

valid = DatasetContainer()
valid.fastMRI(path='path_to_fastMRI_validation', datasetname='fastMRI', dataset_type='training')

for entry in valid:
    train.add_entry(entry)

train, valid = train.split(split=0.95, seed=42)

# Use fastMRI masks
mask_generator = create_mask_for_mask_type(
    "equispaced_fraction",
    center_fractions=[0.08, 0.04],
    accelerations=[4, 8],
    )

# On the fly processing
transforms = torchvision.transforms.Compose([
    DownsampleFOV(k_size=320, i_size=320, complex_support=True, quadratic=True),
    ApplyMaskColumn(mask=mask_generator),
    NormalizeKspace(center_fraction=0.04, complex_support=True),
    lambda x: x.astype(np.complex64),
    ComplexNumpyToTensor(complex_support=False),
    ])


truth_transforms = torchvision.transforms.Compose([
    DownsampleFOV(k_size=320, i_size=320, complex_support=True, quadratic=True),
    NormalizeKspace(center_fraction=0.04, complex_support=True),
    KspaceToImage(norm='ortho', complex_support=True),
    ComplexAbsolute(),  # Better to use numpy complex absolute than fastmri complex absolute
    lambda x: x.astype(np.float32),
    ComplexNumpyToTensor(complex_support=True),
    RSS(),
    ])


training_loader = DatasetLoader(
    datasetcontainer=train,
    train_transforms=transforms,
    truth_transforms=truth_transforms
    )

validation_loader = DatasetLoader(
    datasetcontainer=valid,
    train_transforms=transforms,
    truth_transforms=truth_transforms
    )

# Loss function
loss = [(1, SSIM()), (1, torch.nn.L1Loss())]


loss = MultiLoss(losses=loss)


# Histogram, skille for vev
# Radiomics for skille finstruktur, pyradiomics
metrics = {
    'SSIM': SSIM(),
    'FSSIM': FSSIMLoss(),
    'PSNR': PSNR(),
    'NMSE': NMSE(),
    'MSE': torch.nn.MSELoss(),
    'L1': torch.nn.L1Loss()}

metrics = MultiMetric(metrics=metrics)

path = 'dircn.json'

# 80 million params
model = DIRCN(
    num_cascades=20,
    n=28,
    sense_n=12,
    groups=4,
    sense_groups=1,
    bias=True,
    ratio=1./8,
    dense=True,
    variational=False,
    interconnections=True,
    )


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)

config = ConfigReader(config=path)

train_loader = torch.utils.data.DataLoader(dataset=training_loader,
                                           num_workers=config.num_workers,
                                           batch_size=config.batch_size,
                                           shuffle=config.shuffle)


valid_loader = torch.utils.data.DataLoader(dataset=validation_loader,
                                           num_workers=config.num_workers,
                                           batch_size=config.batch_size,
                                           shuffle=config.shuffle)

trainer = Trainer(
    model=model,
    loss_function=loss,
    metric_ftns=metrics,
    config=config,
    data_loader=train_loader,
    valid_data_loader=valid_loader,
    seed=None,
    # log_step=2500,
    device='cuda:3',
    )

# trainer.resume_checkpoint(
    # resume_model="checkpoint_resume_file",
    # resume_metric='training_statistics_resume_file',
    # )

trainer.train()
