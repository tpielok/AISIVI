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

import matplotlib.pyplot as plt 

def main(config, res_path, log_file):
    # Get device
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

    target = target_distribution[config.target_score](device)

    inp_dim = config.inp_dim
    dim = config.dim
    hidden_dim = config.h_dim

    batch_size = config.batchsize
    batch_size_inner = config.batchsize_inner

    if config.net == "fcn":
        nn_gen = au.sample_simple_fc_net
    elif config.net == "res":
        nn_gen = au.sample_simple_res_net

    model = AISivi(inp_dim, hidden_dim, dim, device, nn_gen, alpha=config.alpha)

    if config.method == "aisivi":
        cnf = generate_cond_real_nvp(config.num_flows, inp_dim, dim, device)

        print("Flow pretraining")

        opt_flow = torch.optim.Adam(cnf.parameters(), lr=config.lr_flow)

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

        scheduler_flow = torch.optim.lr_scheduler.StepLR(opt_flow, step_size=config.gamma_step, gamma=config.gamma)

    print("Model training")

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=config.gamma_step, gamma=config.gamma)

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
            samples = model.sample(config.sampling.num)[0].detach().cpu()
            target.contour_plot(config.bbox, fnet = None, samples=samples,
                                 save_to_path=res_path / Path(f"cp-{(i+1):05d}.png"), quiver = False, t = None)
            
            
            with open(log_file, 'a') as f:
                if config.method == "aisivi":
                    target_samples = target.sample(100000)
                    fDKL = (target.logp(target_samples) - model.log_prob_batch(target_samples, 128, None, 1, False, cnf)).mean()
                    target_samples = target.sample(100000)
                    fDKL2 = (target.logp(target_samples) - model.log_prob_batch(target_samples, 128, None, 1, False, cnf)).mean()
                else:
                    target_samples = target.sample(100000)
                    fDKL = (target.logp(target_samples) - model.log_prob_batch(target_samples, 4096, None, 10, False, None)).mean()
                    target_samples = target.sample(100000)
                    fDKL2 = (target.logp(target_samples) - model.log_prob_batch(target_samples, 4096, None, 10, False, None)).mean()
                f.write(("Epoch [{}/{}], loss: {:.4f}, KL {:.4f}, KL2 {:.4f}\n").format(i, config.num_epochs, loss, fDKL, fDKL2))
                
            samples = model.sample(5000000)[0].detach().cpu()
            plt.hist2d(samples[:, 0], samples[:, 1], bins=300, range=[[-5, 5], [-4, 4]])
            #plt.title("Iteration " + str(i+1) + ", KL: " + str(fDKL.detach().cpu().numpy()))
            plt.title(config.method.upper())
            plt.savefig(res_path / Path(f"hist-{(i+1):05d}.png"))

            torch.save(model, res_path / Path("model.pt"))
            if config.method == "aisivi":
                torch.save(cnf, res_path / Path("flow.pt"))

            torch.save(model.sample(200000)[0].detach().cpu(), res_path / Path("samples.pt"))

if __name__ == '__main__':
    seednow = 2022
    torch.manual_seed(seednow)
    torch.cuda.manual_seed_all(seednow)
    np.random.seed(seednow)
    random.seed(seednow)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--method", type=str, help="aisivi/bsivi", default="aisivi", nargs='?')
    parser.add_argument("--target", type=str, help="banana/x_shaped/multimodal", default="banana", nargs='?')
    args = parser.parse_args()

    config = au.parse_config(Path(args.target) / Path(args.method + ".yml"), argparse.Namespace)

    res_path = (Path('results') / Path(args.target) / Path(args.method))
    res_path.mkdir(parents=True, exist_ok=True)
    log_file = res_path / Path("log.txt")
    log_file.unlink(missing_ok=True)

    config.method = args.method
    main(config, res_path, log_file)