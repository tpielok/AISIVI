# This file is adapted from:
# https://github.com/longinYu/KSIVI/

import torch
from pathlib import Path
import yaml
import numpy as np
import scipy.io

from typing import Optional, Tuple
import itertools
import matplotlib.pyplot as plt

class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs
    
def sample_simple_res_net(inp_dim, hidden_dim, dim, device, global_var=True):
    if global_var:
        return torch.nn.Sequential(torch.nn.Linear(inp_dim, hidden_dim),
                                        torch.nn.ReLU(),
                                        ResNet(torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                                                    torch.nn.ReLU())),
                                        torch.nn.Linear(hidden_dim, dim)
                                        ).to(device)
    else:
        return torch.nn.Sequential(torch.nn.Linear(inp_dim, hidden_dim),
                                        torch.nn.ReLU(),
                                        ResNet(torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                                                    torch.nn.ReLU())),
                                        torch.nn.Linear(hidden_dim, dim*2)
                                        ).to(device)

        
def sample_simple_fc_net(inp_dim, hidden_dim, dim, device, global_var=True):
    if global_var:
        return torch.nn.Sequential(torch.nn.Linear(inp_dim, hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim, hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim, dim)
                                        ).to(device)
    else:
        return torch.nn.Sequential(torch.nn.Linear(inp_dim, hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim, hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim, dim*2)
                                        ).to(device)

        

def parse_config(config, namespace):
    def dict2namespace(config):
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace


    with open(Path("configs") / config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return new_config

def density_estimation(m1, m2):
        x_min, x_max = m1.min(), m1.max()
        y_min, y_max = m2.min(), m2.max()
        X, Y = np.mgrid[x_min : x_max : 100j, y_min : y_max : 100j]                                                     
        positions = np.vstack([X.ravel(), Y.ravel()])                                                       
        values = np.vstack([m1, m2])                                                                        
        kernel = scipy.stats.gaussian_kde(values)                                                                 
        Z = np.reshape(kernel(positions).T, X.shape)
        return X, Y, Z

class EMA:
    def __init__(self, beta, model_params):
        self.beta = beta
        # self.model_params = list(model_params)
        self.shadow_params = [p.clone().detach() for p in model_params]
        self.collected_params = []

    def update_params(self, model_parameters):
        for sp, mp in zip(self.shadow_params, model_parameters):
            sp.data = self.beta * sp.data + (1.0 - self.beta) * mp.data

    def apply_shadow(self, model_parameters):
        for sp, mp in zip(self.shadow_params, model_parameters):
            mp.data.copy_(sp.data)
    
    # for inference
    def store(self, model_parameters):
        self.collected_params = [param.clone() for param in model_parameters]

    def restore(self, model_parameters):
        for c_param, param in zip(self.collected_params, model_parameters):
            param.data.copy_(c_param.data)


def plot_contours(log_prob_func,
                  ax: Optional[plt.Axes] = None,
                  bounds: Tuple[float, float] = (-5.0, 5.0),
                  grid_width_n_points: int = 20,
                  n_contour_levels: Optional[int] = None,
                  log_prob_min: float = -1000.0):
    """Plot contours of a log_prob_func that is defined on 2D"""
    if ax is None:
        fig, ax = plt.subplots(1)
    x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = log_prob_func(x_points).detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)

def plot_marginal_pair(samples: torch.Tensor,
                  ax: Optional[plt.Axes] = None,
                  marginal_dims: Tuple[int, int] = (0, 1),
                  bounds: Tuple[float, float] = (-5.0, 5.0),
                  alpha: float = 0.5):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = torch.clamp(samples, bounds[0], bounds[1])
    samples = samples.cpu().detach()
    ax.plot(samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=alpha)