
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

def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def main(config, res_path):
    # Get device
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

    n = config.dim
    num_comb = int(scipy.special.binom(n+2-1, 2) - n)
    rho_coefs = torch.zeros([num_comb, 2], device=device)

    # Target and data
    target = target_distribution[config.target_score](device)
    data = scipy.io.loadmat('datasets/waveform.mat')
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_train = torch.from_numpy(X_train).to(device).float()
    y_train = torch.from_numpy(y_train).to(device).reshape(-1,1).float()

    scale_sto = X_train.shape[0]/config.sto_batchsize
    baseline_sample = torch.load(Path("datasets/parallel_SGLD_LRwaveform.pt"))

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

            log_p = (target.score(z.detach(), X_train, y_train, scale_sto) * z).sum(dim=1) # surrogate if score of p exists

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
                samples, _ = model.sample(config.sampling.num)
                samples = samples.detach()
                plt.cla()
                c= 0
                for ii in range(n):
                    for jj in range(ii+1,n):
                        #print(torch.corrcoef(sgld_samples[:, (i,j)].t())[0, 1])
                        rho_coefs[c, 0] = torch.corrcoef(baseline_sample[:, (ii,jj)].t())[0, 1]
                        rho_coefs[c, 1] = torch.corrcoef(samples[:, (ii,jj)].t())[0,1]
                        c = c +1
                slope = rho_coefs.prod(dim=1).sum() / (rho_coefs[:, 0]**2).sum() # no intercept
                fig, ax = plt.subplots()
                ax.scatter(rho_coefs[:, 0].cpu().numpy(), rho_coefs[:, 1].cpu().numpy())
                plt.xlim(-0.7, 0.7)
                plt.ylim(-0.7, 0.7)
                add_identity(ax, color='black', ls='--')

                x_vals = np.array(ax.get_xlim())
                y_vals =  slope.cpu().numpy() * x_vals
                ax.plot(x_vals, y_vals, '--')
                plt.title("AISIVI")

                it = j*int(config.num_epochs) + i + 1
                save_to_path = res_path / Path(f"cor-{it:05d}.png") 

                plt.savefig(save_to_path)
                plt.close()


                plt.cla()
                figpos, axpos = plt.subplots(5, 5,figsize = (15,15), constrained_layout=False)
                z = z.detach().cpu().numpy()
                for plotx in range(1,6):
                    for ploty in range(1,6):
                        if ploty != plotx:
                            X1, Y1, Z = au.density_estimation(z[:,plotx], z[:,ploty])
                            axpos[plotx-1,ploty-1].contour(X1, Y1, Z,colors= "#ff7f0e")
                            X1, Y1, Z = au.density_estimation(baseline_sample[:,plotx].cpu().numpy(), baseline_sample[:,ploty].cpu().numpy())
                            axpos[plotx-1,ploty-1].contour(X1, Y1, Z,colors= 'black')
                        else:
                            sns.kdeplot(z[:,plotx], fill=True,color= "#ff7f0e",ax = axpos[plotx-1, ploty-1], label=config.method).set(ylabel=None)
                            sns.kdeplot(baseline_sample[:,plotx].cpu().numpy(),fill=True,color= "black",ax = axpos[plotx-1, ploty-1], label="SGLD").set(ylabel=None)
                            axpos[plotx-1,ploty-1].legend()

                figpos.tight_layout()
                
                save_to_path = res_path / Path(f"sp-{it:05d}.png") 
                
                plt.savefig(save_to_path)
                plt.close()

                torch.save(model, res_path / Path("model.pt"))
                if config.method == "aisivi":
                    torch.save(cnf, res_path / Path("flow.pt"))


if __name__ == '__main__':
    seednow = 2022
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