# General imports
import numpy as np
import torch
import matplotlib.pylab as plt
import os

# DeepMoD functions


from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.model.func_approx import NN
from deepymod.model.constraint import LeastSquares
from deepymod.model.sparse_estimators import Threshold, PDEFIND
from deepymod.training import train
from deepymod.training.sparsity_scheduler import TrainTestPeriodic
from scipy.io import loadmat
from deepymod.model.library import Library1D


import torch
from torch.autograd import grad
from itertools import combinations
from functools import reduce
from typing import Tuple
from deepymod.utils.types import TensorList
from deepymod import Library

from scipy.integrate import odeint

# Settings for reproducibility
for i in range(10):
    np.random.seed(40+i)
    torch.manual_seed(0+i)

    # Configuring GPU or CPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)

    def dU_dt_true(U):
        """
        returns the right hand side of the differential equation"""
        sigma = 10
        rho = 28
        beta = 8/3
        return [sigma*(U[1]-U[0]), U[0]*(rho - U[2])-U[1], U[0]*U[1] - beta*U[2]]


    def dU_dt_sin(U, t):
        """
        returns the right hand side of the differential equation"""
        return dU_dt_true(U)

    dt = 0.001
    x0_train = [-8, 8, 27]
    def create_data(U0=x0_train, ts=np.arange(0, 100, dt)):
        """
        Creates data which is the solution of the simple ODE system example.
        the output has torch.float32 format.
        
        Args: 
            U0: Initial condition
            ts: Time points to evaluate the ODE at.
        """
        Y = torch.from_numpy(odeint(dU_dt_sin, U0, ts)).float()
        T = torch.from_numpy(ts.reshape(-1, 1)).float()
        return T, Y

    def custom_normalize(feature):
            """minmax all features by their absolute maximum
            Args:
                feature (torch.tensor): data to be minmax normalized
            Returns:
                (torch.tensor): minmaxed data"""
            return (feature/feature.abs().max(axis=0).values)

    dataset = Dataset(
        create_data,
        subsampler=Subsample_random,
        subsampler_kwargs={"number_of_samples": 100000},
        preprocess_kwargs={"noise_level": 0,  
            "normalize_coords": True,
            "normalize_data": True,},
        apply_normalize=custom_normalize,
        device=device
    )
    dataset.data.shape

    # see deepymod.data.base.get_train_test_loader for definition of `get_train_test_loader` function.
    # the shuffle is completely random mixing latter and earlier times
    train_dataloader, test_dataloader = get_train_test_loader(dataset, train_test_split=0.8)

    library = Library1D(poly_order=1, diff_order=0) 
    estimator = Threshold(0.05)
    constraint = LeastSquares()
    sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=200, delta=1e-5)
    network = NN(1, [30, 30, 30, 30], 3)
    model = DeepMoD(network, library, estimator, constraint).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3
    )

    os.system(f"rm -rf ./data/deepymod/Lorenz63_{i}/")
    foldername = f"./data/deepymod/Lorenz63_{i}/"
    train(
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        sparsity_scheduler,
        log_dir=foldername,
        max_iterations=100000,
        delta=1e-3,
        patience=100,
    )

    print(model.sparsity_masks)
    print(model.constraint_coeffs())