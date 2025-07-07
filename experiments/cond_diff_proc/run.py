
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import argparse

from aisivi.model import AISivi
import aisivi.utils as au
from aisivi.targets import target_distribution
from aisivi.cnf import *
from aisivi.cnf import generate_cond_real_nvp

from tqdm import tqdm
from pathlib import Path

import random
import scipy.io

import matplotlib.pyplot as plt
import seaborn as sns

def main(config, res_path):
    # Get device
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

    # Target and data
    target = target_distribution[config.target_score](num_interval = config.num_interval, num_obs = config.num_obs, beta = config.beta, T = config.T, sigma = config.sigma, device=device)

    inp_dim = config.inp_dim
    dim = config.dim
    hidden_dim = config.h_dim

    batch_size = config.batchsize
    batch_size_inner = config.batchsize_inner

    if config.net == "fcn":
        nn_gen = au.sample_simple_fc_net
    elif config.net == "res":
        nn_gen = au.sample_simple_res_net

    if config.method == "aisivi":
        cnf = generate_cond_real_nvp(config.num_flows, inp_dim, dim, device)
        opt_flow = torch.optim.Adam(cnf.parameters(), lr=config.lr_flow, weight_decay=config.weight_decay_flow)
        scheduler_flow = torch.optim.lr_scheduler.StepLR(opt_flow, step_size=config.gamma_step, gamma=config.gamma)

    model = AISivi(inp_dim, hidden_dim, dim, device, nn_gen, alpha=config.alpha)


    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=config.gamma_step, gamma=config.gamma)

    for j in tqdm(range(config.outer_loop)):
        if config.method == "aisivi":
            
            print("Flow training")

            for i in (pbar := tqdm(range(config.num_epochs_flow))):
                opt_flow.zero_grad()

                with torch.no_grad():
                    eps = model.sample_eps(batch_size)
                    z, _ = model.sample(batch_size, model(eps))

                flow_loss = -cnf.log_prob(eps, context=z).mean()

                if ~(torch.isnan(flow_loss) | torch.isinf(flow_loss)):
                    flow_loss.backward()
                    opt_flow.step()

                    if i % 10 == 0:
                        pbar.set_description("{:.5f}".format(flow_loss))

        print("Model training")

        
        for i in (pbar := tqdm(range(config.num_epochs))):

            with torch.no_grad():
                eps = model.sample_eps(batch_size)
                z, _ = model.sample(batch_size, model(eps))

            if config.method == "aisivi":
                opt_flow.zero_grad()
                flow_loss = -cnf.log_prob(eps, context=z).mean()

                if ~(torch.isnan(flow_loss) | torch.isinf(flow_loss)):
                    flow_loss.backward()
                    opt_flow.step()

            opt.zero_grad()
            
            if config.method == "aisivi":
                z, _ = model.sample(batch_size, model(eps))
                _, grad_log_q = model.log_prob_batch(z, batch_size_inner, None, 
                                                config.num_inner_batches, encoder=cnf, compute_grad=True)
            else:
                z, params = model.sample(batch_size, model(eps))
                _, grad_log_q = model.log_prob_batch(z, batch_size_inner, params, 
                                                config.num_inner_batches, encoder=None, compute_grad=True)

            log_q = (grad_log_q.detach() * z).sum(dim=1) # surrogate for path gradient computation

            log_p = (target.score(z.detach()) * z).sum(dim=1) # surrogate if score of p exists

            loss = (log_q - log_p).mean()
            
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                opt.step()
                scheduler.step()
                if config.method == "aisivi":
                    scheduler_flow.step()

                if i % 10 == 0:
                    pbar.set_description("{:.5f}".format(loss))
            
            if (i+1) % config.visual_time == 0:
                X = model.sample(config.sampling.num)[0]
                target.trace_plot(X, figpath = res_path, figname = f"sp-{(j*config.num_epochs + i + 1):05d}.png")

            if (i+1) % config.sampling_time == 0:
                torch.save(model, res_path / f"model-{(j*config.num_epochs + i + 1):05d}.pt")

                if config.method == "aisivi":
                    torch.save(cnf, res_path / f"fmodel-{(j*config.num_epochs + i + 1):05d}.pt")

                torch.save(X, res_path / f"samples-{(j*config.num_epochs + i + 1):05d}.pt")

if __name__ == '__main__':
    seednow = 2023
    torch.manual_seed(seednow)
    torch.cuda.manual_seed_all(seednow)
    np.random.seed(seednow)
    random.seed(seednow)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--method", type=str, help="aisivi/bsivi", default="aisivi", nargs='?')
    args = parser.parse_args()

    config = au.parse_config(Path(args.method + ".yml"), argparse.Namespace)

    res_path = (Path('results') / Path(args.method))
    res_path.mkdir(parents=True, exist_ok=True)

    config.method = args.method
    main(config, res_path)