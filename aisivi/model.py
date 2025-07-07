import torch
import torch.distributions as D
import torch.nn as nn

from torch.nn.functional import softplus

from aisivi.utils import sample_simple_res_net

c = 1e-10

class AISivi(nn.Module):

    def __init__(self, inp_dim, hidden_dim, dim, device, nn_gen_fn=sample_simple_res_net, alpha=1.0, global_var=True, var_anneal=False):
        
        super().__init__()
        
        self.device = device
        self.dim    = dim

        self.var_anneal = var_anneal

        self.inp_dim = inp_dim
        if var_anneal:
            self.nn = nn_gen_fn(self.inp_dim + 1, hidden_dim, dim, device, global_var)
        else:
            self.nn = nn_gen_fn(self.inp_dim, hidden_dim, dim, device, global_var)

        self.alpha = alpha

        self.global_var = global_var
        if global_var:
            self.logexpm1_var = nn.Parameter(torch.ones(dim, device=device), requires_grad=True)

    def forward(self, x, t=None):
        if self.var_anneal:
            if x.dim() == 2:
                x = torch.hstack([x, t])
            else:
                #x = torch.cat([x, t[..., None].repeat((1, x.shape[1], 1))], dim=2)
                x = torch.cat([x, t[..., None].expand(-1, x.shape[1], -1)], dim=2)

        return self.nn(x)
    
    def sample_eps(self, k, m=None, rep=True):
        if m is None:
            return torch.rand(k, self.inp_dim, device=self.device)
        elif rep:
            return torch.rand(1, k, self.inp_dim, device=self.device).expand(m, -1, -1)
        else:
            return torch.rand(m, k, self.inp_dim, device=self.device)
    
    def sample_params(self, k, t=None, rep=True, for_each_obs=True):
        if t is None:
            return self.forward(self.sample_eps(k), t=None)
        else:
            if for_each_obs:
                return self.forward(self.sample_eps(k, m=t.shape[0], rep=rep), t=t)
            else:
                return self.forward(self.sample_eps(k), t=t)
        
    
    def sample(self, m, params=None, t=None):
        dim = self.dim
        
        if params is None:
            params = self.sample_params(m, t=t, for_each_obs=False)
            
        if self.global_var:
            logexpm1_var = self.logexpm1_var
        else:
            logexpm1_var = params[..., dim:]

        dist = D.Normal(params[..., :dim], c+self.alpha*softplus(logexpm1_var))
                
        return dist.rsample([1])[0, ...], params
    
    def log_prob_z_eps(self, z, params, for_each_obs=True):
        # z:        n*d
        # params:   (n)*k*(d|d/2) 

        dim = self.dim
        params_det = params.detach()

        if params_det.dim() == 2 and for_each_obs:
            params_det = params_det[None, ...]

        if self.global_var:
            logexpm1_var = self.logexpm1_var
        else:
            logexpm1_var = params[..., dim:]
        
        dist = D.Normal(params_det[..., :dim], c+ self.alpha*softplus(logexpm1_var))
        
        if for_each_obs:
            return dist.log_prob(z[:, None, ...]).sum(dim=-1)
        else:
            return dist.log_prob(z).sum(dim=-1)

    def log_prob(self, z, k, params=None, encoder=None, compute_grad = False, t=None, rep=True, for_each_obs=True):
        dim = self.dim

        if encoder is None:
            with torch.no_grad():
                if params is None:
                    params = self.sample_params(k, t, rep=rep, for_each_obs=for_each_obs)
                else:
                    params = params.detach()

                    if t is not None and params.dim() == 2 and for_each_obs:
                        params = params[:, None, :]

                    if params.shape[-2] < k:
                        params = torch.cat([params, self.sample_params(k - params.shape[-2], t, rep=rep, for_each_obs=
                                                                       for_each_obs)], dim=-2)  

            if not compute_grad:
                return self.log_prob_z_eps(z, params).logsumexp(dim=-1) - torch.tensor(k).log()
            else:
                z.requires_grad = True

                lp = self.log_prob_z_eps(z, params).logsumexp(dim=-1) - torch.tensor(k).log()
                return lp, torch.autograd.grad(lp.sum(), z)[0]
        else:
            m = z.shape[0]
            eps_z, lps = encoder.sample(k, z)
            #z_rep = z.repeat_interleave(k, 0)
            
            
            #while True:
            #    eps_z, lps = encoder.sample(m*k, z_rep)
            #    if not eps_z.isnan().any():
            #        break
            #m = context.shape[0]

            params_z = self.forward(eps_z, t)

            if compute_grad:
                z.requires_grad = True
                q_lps = (self.log_prob_z_eps(z, params_z.reshape([m, k, params_z.shape[-1]])) - lps.reshape(m, k)).logsumexp(dim=-1) - torch.tensor(k).log()

                return q_lps.detach(), torch.autograd.grad(q_lps.sum(), z)[0].detach()
            else:
                return (self.log_prob_z_eps(z, params_z.reshape([m, k, params_z.shape[-1]])) - lps.reshape(m, k)).logsumexp(dim=-1) - torch.tensor(k).log() 

      
    def log_prob_batch(self, z, k, params=None, num_batches=1, compute_grad=False, encoder=None, t=None, rep=True,
                       for_each_obs=True):
        z_det = z.detach()
        log_q = None
        if compute_grad:
            grad_log_q = None

        for j in torch.arange(num_batches).to(self.device):
            if compute_grad:
                z_det.requires_grad = True

            if j == 0:
                if compute_grad:
                    log_q_inner, grad_log_q_inner = self.log_prob(z_det, k, params, encoder=encoder, compute_grad=compute_grad, t=t, rep=rep,
                                                                  for_each_obs=for_each_obs)
                    grad_log_q = grad_log_q_inner.detach()
                else:
                    log_q_inner = self.log_prob(z_det, k, encoder=encoder, compute_grad=compute_grad, t=t, rep=rep, for_each_obs=
                                                for_each_obs)  

                log_q = log_q_inner.detach()
                params = None
            else:
                if compute_grad:
                    log_q_inner, grad_log_q_inner = self.log_prob(z_det, k, encoder=encoder, compute_grad=compute_grad, t=t, rep=rep,
                                                                  for_each_obs=for_each_obs)    
                else:
                    log_q_inner = self.log_prob(z_det, k, encoder=encoder, compute_grad=compute_grad, t=t, rep=rep,
                                                for_each_obs=for_each_obs)  

                log_q_inner = log_q_inner.detach()

                log_q_sum_old = log_q + j.log()
                log_q_sum_new = torch.logaddexp(log_q_sum_old, log_q_inner)

                if compute_grad:
                    grad_log_q = (log_q_sum_old - log_q_sum_new).exp()[:, None]*grad_log_q + (log_q_inner - log_q_sum_new).exp()[:, None]*grad_log_q_inner

                log_q = log_q_sum_new - (j+1).log()
        
        if compute_grad:
            return log_q, grad_log_q
        else:
            return log_q
