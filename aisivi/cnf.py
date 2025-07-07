# This file is adapted from:
# https://https://github.com/VincentStimper/normalizing-flows

import normflows as nf
import torch
import numpy as np
from torch.nn.functional import softplus

min_s = 1e-7

class CondMaskedAffineFlow(nf.flows.Flow):
    """RealNVP as introduced in [arXiv: 1605.08803](https://arxiv.org/abs/1605.08803)

    Masked affine flow:

    ```
    f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    ```

    - class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    - NICE is AffineFlow with only shifts (volume preserving)
    """

    def __init__(self, b, t=None, s=None):
        """Constructor

        Args:
          b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
          t: translation mapping, i.e. neural network, where first input dimension is batch dim, if None no translation is applied
          s: scale mapping, i.e. neural network, where first input dimension is batch dim, if None no scale is applied
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)

        if s is None:
            self.s = torch.zeros_like
        else:
            self.add_module("s", s)

        if t is None:
            self.t = torch.zeros_like
        else:
            self.add_module("t", t)

    def forward(self, z, context):
        z_masked = self.b * z
        scale = self.s(torch.hstack([z_masked, context]))
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(torch.hstack([z_masked, context]))
        trans = torch.where(torch.isfinite(trans), trans, nan)
        #z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        trafo_scale = softplus(scale) + min_s
        z_ = z_masked + (1 - self.b) * (z * trafo_scale + trans)
        #log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        log_det = torch.sum((1 - self.b) * trafo_scale.log(), dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z, context):
        z_masked = self.b * z
        scale = self.s(torch.hstack([z_masked, context]))
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(torch.hstack([z_masked, context]))
        trans = torch.where(torch.isfinite(trans), trans, nan)
        #z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        trafo_scale = softplus(scale) + min_s
        z_ = z_masked + (1 - self.b) * (z - trans) / trafo_scale
        #log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        log_det = -torch.sum((1 - self.b) * trafo_scale.log(), dim=list(range(1, self.b.dim())))
        return z_, log_det
        
    
class ActNorm(nf.flows.affine.coupling.AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done_cpu = torch.tensor(0.0)
        self.register_buffer("data_dep_init_done", self.data_dep_init_done_cpu)

    def forward(self, z, context=None):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.0:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (
                -z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)
            ).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().forward(z)

    def inverse(self, z, context=None):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().inverse(z)

class Logit(nf.flows.Flow):
    """Logit mapping of image tensor, see RealNVP paper

    ```
    logit(alpha + (1 - alpha) * x) where logit(x) = log(x / (1 - x))
    ```

    """

    def __init__(self, alpha=0.05):
        """Constructor

        Args:
          alpha: Alpha parameter, see above
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, z, context=None):
        beta = 1 - 2 * self.alpha
        sum_dims = list(range(1, z.dim()))
        ls = torch.sum(torch.nn.functional.logsigmoid(z), dim=sum_dims)
        mls = torch.sum(torch.nn.functional.logsigmoid(-z), dim=sum_dims)
        log_det = -np.log(beta) * np.prod([*z.shape[1:]]) + ls + mls
        z = (torch.sigmoid(z) - self.alpha) / beta
        return z, log_det

    def inverse(self, z, context=None):
        beta = 1 - 2 * self.alpha
        z = self.alpha + beta * z
        logz = torch.log(z)
        log1mz = torch.log(1 - z)
        z = logz - log1mz
        sum_dims = list(range(1, z.dim()))
        log_det = (
            np.log(beta) * np.prod([*z.shape[1:]])
            - torch.sum(logz, dim=sum_dims)
            - torch.sum(log1mz, dim=sum_dims)
        )
        return z, log_det
    
class ConditionalNormalizingFlowAI(nf.ConditionalNormalizingFlow):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def sample(self, k, context):
        m = context.shape[0]
        z_rep = context.repeat_interleave(k, 0)

        while True:
            eps_z, lps = super().sample(m*k, z_rep)
            if not eps_z.isnan().any():
                break
                
        return eps_z, lps
    
def generate_cond_real_nvp(K, latent_size, context_size, device, act_norm=True):
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([latent_size + context_size, 2 * (latent_size + context_size), latent_size], init_zeros=True, leaky=0.0)
        t = nf.nets.MLP([latent_size + context_size, 2 * (latent_size + context_size), latent_size], init_zeros=True, leaky=0.0)
        if i % 2 == 0:
            flows += [CondMaskedAffineFlow(b, t, s)]
        else:
            flows += [CondMaskedAffineFlow(1 - b, t, s)]
        if act_norm:
            flows += [ActNorm(latent_size)]
    flows += [Logit(0.0)]

    # Set q0
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)

    # Construct flow model
    nfm = ConditionalNormalizingFlowAI(q0, flows, None)
    #nfm = nf.ConditionalNormalizingFlow(q0, flows, None)

    nfm = nfm.to(device)

    # Initialize ActNorm
    if act_norm:
        num_samples=2 ** 7
        z, _ = nfm.sample(num_samples, context=torch.randn([num_samples, context_size], device=device))

    return nfm