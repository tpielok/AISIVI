# This file is adapted from:
# https://github.com/longinYu/KSIVI/blob/main/models/target_models.py

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as st

from pathlib import Path


class Toy_2D(object):
    def __init__(self, name):
        self.name = name
    def logp(self, X):
        pass
    def score(self, X):
        pass
    def contour_plot(self, bbox, fnet = None, samples=None, save_to_path=None, quiver = True, t = None):
        plt.cla()
        fig, ax = plt.subplots(figsize=(5, 5))
        xx, yy = np.mgrid[bbox[0]:bbox[1]:100j, bbox[2]:bbox[3]:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = -np.log(-np.reshape(self.logp(torch.Tensor(positions.T).to(self.device)).cpu().numpy(), xx.shape))
        if samples is None:
            samples = self.sample(10000).cpu().numpy()
        
        cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
        
        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
        cfset = ax.contour(xx, yy, f, levels = 11, colors='black')
        ax.plot(samples[:, 0], samples[:,1], '.', markersize= 2, color='blue', alpha=0.05)
        if quiver:
            cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
            scores = np.reshape(fnet(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(), cpositions.T.shape)
            ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        if t:
            ax.set_title("t = {}".format(t), fontsize = 30, y=1.04)
        else:
            ax.set_title(f"{self.name}", fontsize = 20, y=1.04)
        if save_to_path is not None:
            plt.savefig(save_to_path, bbox_inches='tight')
            plt.close()

class Neal():
    name = "neal"
    def __init__(self, device, d):
        self.device = device

        self.d = d

    def sample(self, num_samples):
        samples = torch.randn([num_samples, self.d], device=self.device) 
        samples[:, 0] *=  3.0
        samples[:, 1:] = samples[:, 1:] * (samples[:, 0:1] / 2).exp()

        return samples
    
    def logp(self, samples):
        return torch.distributions.Normal(0, (samples[:, 0:1] / 2).exp()).log_prob(samples[:, 1:]).sum(dim=1) \
            + torch.distributions.Normal(0, torch.tensor(3, device=samples.device)).log_prob(samples[:, 0])

    def score(self, x):
        s = torch.empty(x.shape, device=x.device)
        s[:, 1:] = -x[:, 1:] * (-x[:, 0:1]).exp()
        s[:, 0] =  -x[:, 0] / 9.0 - (s[:, 1:]*x[:, 1:]/2 + 0.5).sum(dim=1)

        return s

class Banana_shape(Toy_2D):
    name = "banana_shape"
    def __init__(self, device):
        self.device = device
        self.dist = torch.distributions.MultivariateNormal(torch.zeros(2, device=device), torch.tensor([[1, 0.9],[0.9, 1]], device=device))

        super().__init__("Banana")
    def logp(self, X):
        Y = torch.stack((X[:, 0], X[:, 0]**2 + X[:, 1] + 1), 1)
        sigmasqinv = torch.tensor([[1.0, -0.9], [-0.9, 1.0]]).to(self.device)/0.19
        return -0.5 * 2 * np.log(2 * np.pi) - 0.5 * np.log(0.19) - 0.5 * torch.matmul(torch.matmul(Y[:,None,:],sigmasqinv), Y[:,:,None]).squeeze(-1)

    def score(self, X):
        Y = torch.matmul(torch.stack((X[:, 0], X[:, 0]**2 + X[:, 1] + 1), 1),torch.tensor([[1.,-0.9],[-0.9,1.]]).to(self.device))
        return -torch.stack((Y[:,0] + 2 * X[:,0] * Y[:,1], Y[:, 1]),1)/0.19
    
    def sample(self, num_samples):
        Y = self.dist.sample([num_samples])
        Y[:, 1] += Y[:, 0]**2 + 1
        return -Y
    
class X_shaped(Toy_2D):
    name = "x_shaped"
    def __init__(self, device):
        self.device = device
        super().__init__("X shape")
        self.dist1 = torch.distributions.MultivariateNormal(torch.zeros(2, device=device), torch.tensor([[2, 1.8],[1.8, 2]], device=device))
        self.dist2 = torch.distributions.MultivariateNormal(torch.zeros(2, device=device), torch.tensor([[2, -1.8],[-1.8, 2]], device=device))

    def logp(self, X):
        sigmasqinv_0 = torch.tensor([[2., -1.8], [-1.8, 2.]]).to(self.device) / 0.76
        sigmasqinv_1 = torch.tensor([[2., 1.8], [1.8, 2.]]).to(self.device) / 0.76
        return -0.5 * 2 * np.log(2 * np.pi) - 0.5 * np.log(0.76 * 4) + torch.logsumexp(torch.stack(
            (-1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_0), X[:,:,None]).squeeze(-1),
            -1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_1), X[:,:,None]).squeeze(-1)),1
            ), dim = 1)
    def score(self, X):
        sigmasqinv_0 = torch.tensor([[2., -1.8], [-1.8, 2.]]).to(self.device) / 0.76
        sigmasqinv_1 = torch.tensor([[2., 1.8], [1.8, 2.]]).to(self.device) / 0.76

        Y = F.softmax(torch.stack(
            (-1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_0), X[:,:,None]).squeeze(-1),
            -1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_1), X[:,:,None]).squeeze(-1)),1
            ), dim = 1)
    
        return -Y[:,0] * torch.matmul(sigmasqinv_0, X[:,:,None]).squeeze(-1) - Y[:,1] * torch.matmul(sigmasqinv_1, X[:,:,None]).squeeze(-1)
    
    def sample(self, num_samples):
        res = self.dist1.sample([num_samples])
        idx = torch.rand(num_samples).to(self.device) < 0.5
        res[idx] = self.dist2.sample([idx.sum()])
        return res

    
class Multimodal(Toy_2D):
    name = "multimodal"
    def __init__(self, device):
        self.device = device
        self.dist1 = torch.distributions.MultivariateNormal(torch.tensor([-2., 0], device=device), torch.tensor([[1., 0],[0, 1]], device=device))
        self.dist2 = torch.distributions.MultivariateNormal(torch.tensor([2, 0.], device=device), torch.tensor([[1., 0],[0, 1]], device=device))
        super().__init__("Multimodal")
    def logp(self, X):
        means = torch.tensor([[2.0,0.0],[-2.0,0.0]]).to(self.device)
        return -0.5 * 2 * np.log(2 * np.pi) - np.log(2.0) + torch.logsumexp(
            -torch.sum((X.unsqueeze(1) - means.unsqueeze(0))**2, dim=-1)/2./1**2
            , dim = 1)
    def score(self, X):
        Y = F.softmax(torch.stack(
            (-1/2 * ((X[:, 0] + 2)**2 + X[:, 1]**2),
            -1/2 * ((X[:, 0] - 2)**2 + X[:, 1]**2)),1
        ),dim=1)
        return - torch.stack((Y[:,0] * (X[:, 0] + 2) + Y[:,1] * (X[:, 0] - 2), X[:, 1]),1)
    
    def sample(self, num_samples):
        res = self.dist1.sample([num_samples])
        idx = torch.rand(num_samples).to(self.device) < 0.5
        res[idx] = self.dist2.sample([idx.sum()])
        return res

    
class LRwaveform(object):
    name = "LRwaveform"
    def __init__(self, device, alpha = 0.01):
        self.device = device
        self.alpha = alpha

    def logp(self, Z, batchdataset, batchlabel, scale_sto = 1):
        """
        output: the \E_{Y|X}\log \E p(Y|X,z), as the test log ll
        Z: the target inference parameters, shape = [T, (x_dim + 1)]
        batchdataset: the batch bataset with shape = [batchsize, (x_dim + 1)]
        batchlabel: onehot for the the label corresponding to the batchdataset, shape = [n,1]
        scale_sto: num_datasets/batchsize
        """
        B = Z.shape[0]
        W = Z
        inner_prod = torch.mm(batchdataset, W.t())
        logpy_xz = (batchlabel.reshape(-1,1) * inner_prod + F.logsigmoid(-inner_prod))
        return (torch.logsumexp(logpy_xz, dim=1).mean(0) - np.log(B))

    def score(self, Z, batchdataset, batchlabel, scale_sto):
        """
        INPUT:
        Z: the target inference parameters, shape = [batch, (x_dim + 1)]
        batchdataset: the batch bataset with shape = [batchsize, (x_dim + 1)]
        batchlabel: the label corresponding to the batchdataset, shape = [n,1]
        scale_sto: num_datasets/batchsize
        OUTPUT:
        -Z + \nabla_Z\log p(Y|X,Z), where p(y_i = 1|x_i,w) = sigmoid(Z\cdot x_i)
        """
        # batchlabel[batchlabel == -1] = 0
        W = Z
        YX = torch.mm(batchlabel.reshape(-1,1).t(), batchdataset)
        inner_prod = torch.mm(batchdataset, W.t())
        score_W = -W * self.alpha + (YX - torch.sum(torch.sigmoid(inner_prod).unsqueeze(2) * batchdataset.unsqueeze(1), dim=0)) * scale_sto
        return score_W

class Langevin_post(object):
    def __init__(self, num_interval = 100, num_obs = 20, beta=10.0, T=1.0, sigma=0.1, device = 'cpu') -> None:
        self.beta = beta
        self.sigma = sigma
        self.T = T
        self.dt = T/num_interval
        self.dim = num_interval
        self.device=device
        self.u_step = int(num_interval/num_obs)
        self.num_obs = num_obs
        self.upper_mask = torch.triu(torch.ones((self.dim, self.dim), device=device)).contiguous().bool()
        self.upper1_mask = (1-torch.triu(torch.ones((self.dim, self.dim), device=device)).transpose(0,1)).contiguous().bool()
        self.u_mask = (torch.arange(1, self.dim+1) % self.u_step == 0)
        
        torch.manual_seed(2022)
        # xs = torch.load('gaussians.pt', map_location=self.device)
        # xs = torch.randn((self.dim, 1), device=self.device)
        xs = torch.randn((self.dim, 1))
        u = torch.zeros((1, ))
        us = []
        for i in range(self.dim):
            u = u + self.beta * u * (1-u**2) * self.dt + xs[i] * np.sqrt(self.dt)
            us.append(u)
        self.u = torch.tensor(us).to(self.device)
        us = (torch.stack(us).T)[:, self.u_step-1::self.u_step]
        noise = torch.randn_like(us)
        
        data = us + noise * self.sigma 
        self.data = data.to(self.device)
        self.xs = xs.to(self.device)
        self.us = us.to(self.device)

    def logp(self, us):
        us_mean = us[:,:-1] + self.beta * (us[:,:-1] - us[:,:-1]**3) / (1+us[:,:-1]**2) * self.dt
        us_mean_pad = torch.concatenate([torch.zeros((us.shape[0],1), device=self.device), us_mean], dim=-1) 
        logp = - torch.sum((us-us_mean_pad)**2/(2*self.dt), dim=-1) - torch.sum((us[:,None, self.u_mask]-self.data[None,:,:])**2/(2*self.sigma**2), dim=(-1,-2))
        return logp
    
    def score(self, us):
        us_mean = us[:,:-1] + self.beta * (us[:,:-1] - us[:,:-1]**3)/(1 + us[:,:-1]**2) * self.dt
        us_mean_pad = torch.concatenate([torch.zeros((us.shape[0],1), device=self.device), us_mean], dim=-1)
        score_ll_part = - torch.sum((us[:,None, self.u_mask] - self.data[None,:,:])/(self.sigma**2), dim=1)
        score_ll = torch.zeros_like(us, device=self.device)
        score_ll[:, self.u_mask] = score_ll_part

        score_prior_1 = - (us-us_mean_pad)/(self.dt)
        score_prior_2 = - torch.concatenate([(us_mean-us[:,1:])/(self.dt) * (1 - self.beta*self.dt-self.beta*self.dt*2*(us[:,:-1]**2-1)/(us[:,:-1]**2+1)**2), torch.zeros((us.shape[0],1), device=self.device)], dim=-1)
        return (score_prior_1 + score_prior_2) + score_ll
    
    def trace_plot(self, u, figpath=None, figname=None, figtitle = ""):
        u = u.detach().cpu().numpy()
       
        u_mean = u.mean(0)
        low_CI_bound, high_CI_bound = st.t.interval(0.95,len(u_mean),loc=u_mean,scale=np.std(u,0))
        u_true = self.u.detach().cpu().numpy().flatten()

        t = np.arange(self.dt, self.T + self.dt, self.dt)
        plt.plot(t, u_true, color='magenta', label='true path')
        plt.plot(t, u_mean, color='blue', label='sample path')
        plt.plot(t, low_CI_bound, color='black', linewidth=1.0)
        plt.plot(t, high_CI_bound, color='black', linewidth=1.0)
        plt.fill_between(t,
                         low_CI_bound,
                         high_CI_bound,
                         facecolor='aqua',
                         alpha=0.3,
                         label='confidence interval')
        obs_interval = self.T / self.num_obs
        plt.scatter(np.arange(obs_interval, self.T + obs_interval,
                                obs_interval),
                    self.data.detach().cpu().numpy(),
                    color='r',
                    marker='.',
                    linewidth=0.5,
                    label='observation')

        plt.legend()
        plt.grid('on')
        plt.title(figtitle)
        plt.tight_layout()
        plt.savefig(figpath / Path(figname), dpi=600)
        plt.close()

    def trace_subplot(self, u, ax, figtitle = ""):
        u = u.detach().cpu().numpy()
       
        u_mean = u.mean(0)
        low_CI_bound, high_CI_bound = st.t.interval(0.95,len(u_mean),loc=u_mean,scale=np.std(u,0))
        u_true = self.u.detach().cpu().numpy().flatten()

        t = np.arange(self.dt, self.T + self.dt, self.dt)
        ax.plot(t, u_true, color='#E74C3C', label='true path')
        ax.plot(t, u_mean, color='#3498DB', label='sample path')
        ax.plot(t, low_CI_bound, color='black', linewidth=1.0)
        ax.plot(t, high_CI_bound, color='black', linewidth=1.0)
        ax.fill_between(t,
                         low_CI_bound,
                         high_CI_bound,
                         facecolor='#2C3E50',
                         alpha=0.3,
                         label='confidence interval')
        obs_interval = self.T / self.num_obs
        ax.scatter(np.arange(obs_interval, self.T + obs_interval,
                                obs_interval),
                    self.data.detach().cpu().numpy(),
                    color='#2ECC71',
                    edgecolors='black',
                    marker='.',
                    linewidth=1.0,
                    label='observation')

        #plt.legend()
        ax.grid('on')
        ax.set_title(figtitle)

        
class Bnn(object):
    name = "Bnn"
    def __init__(self, device, d, n_hidden = 50, loglambda = 0, loggamma = 0):
        self.device = device
        self.n_hidden = n_hidden
        self.d = d
        self.dim_vars = (self.d + 1) * self.n_hidden + (self.n_hidden + 1) + 2
        self.dim_wb = self.dim_vars - 2
        self.loggamma = loggamma
        self.loglambda = loglambda
    def logp(self, Z, batchdataset, batchlabel, scale_sto = 1, max_param = 50.0):
        """
        return the log posterior distribution \log P(W|Y,X).
        """
        log_gamma = self.loggamma * torch.ones(Z.size(0)).to(self.device)
        log_lambda = self.loglambda * torch.ones(Z.size(0)).to(self.device)
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        lambda_ = torch.exp(log_lambda).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-1].reshape(-1,1) # [B, 1]
        dnn_predict = (torch.matmul(torch.max(torch.matmul(batchdataset, W1) + b1[:,None,:], torch.tensor([0.0]).to(self.device)), W2) + b2[:,None,:])   # [B, n, 1]
        log_lik_data = -0.5 * batchdataset.shape[0] * (np.log(2*np.pi) - log_gamma) - (gamma_/2) * torch.sum(((dnn_predict-batchlabel).squeeze(2))**2, 1)
        log_prior_w = -0.5 * self.dim_wb * (np.log(2*np.pi) - log_lambda) - (lambda_/2)*((W1**2).sum((1,2)) + (W2**2).sum((1,2)) + (b1**2).sum(1) + (b2**2).sum(1))
        return (log_lik_data * scale_sto + log_prior_w)
        
    def score(self, Z, batchdataset, batchlabel, scale_sto = 1, max_param = 50.0):
        """
        return the score function of posterior distribution \nabla \log P(W|Y,X).
        """
        batch_Z = Z.shape[0]
        num_data = batchdataset.shape[0]
        log_gamma = self.loggamma * torch.ones((batch_Z,1)).to(self.device) # [B, 1]
        log_lambda = self.loglambda * torch.ones((batch_Z,1)).to(self.device)
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        lambda_ = torch.exp(log_lambda).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-1].reshape(-1,1) # [B, 1]

        dnn_onelinear = torch.matmul(batchdataset, W1) + b1[:,None,:]
        dnn_relu_onelinear = torch.max(dnn_onelinear, torch.tensor([0.0]).to(self.device))
        dnn_grad_relu = (torch.sign(dnn_onelinear) + 1)/2 # shape = [B, n, hidden]
        dnn_predict = (torch.matmul(dnn_relu_onelinear, W2) + b2[:,None,:]) # shape = [B,n,1]
        nabla_predict_b1 = dnn_grad_relu * W2.transpose(1,2) # [B, n, hidden]
        nabla_predict_W1 = nabla_predict_b1[:,:,None,:] * batchdataset[None,:,:,None] # [B,n,d, hidden] 
        nabla_predict_W2 = dnn_relu_onelinear # [B,n, hidden]
        nabla_predict_b2 = torch.ones_like(dnn_predict).to(self.device) # [B,n,1]

        nabla_predict_wb = torch.cat((nabla_predict_W1.reshape(batch_Z, num_data, -1), nabla_predict_b1, nabla_predict_W2, nabla_predict_b2),dim=2)
        nabla_wb = scale_sto * gamma_ * ((batchlabel - dnn_predict) * nabla_predict_wb).sum(1) - lambda_ * Z
        return nabla_wb      # shape = [B, self.dim_vars]
    def rmse_llk(self, Z, batchdataset, batchlabel, mean_y_train, std_y_train, max_param = 50.0):
        """
        return the test RMSE and test log-likelihood of posterior distribution \nabla \log P(W|Y,X).
        """
        log_gamma = self.loggamma * torch.ones((Z.size(0),1)).to(self.device) # [B, 1]
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-1].reshape(-1,1) # [B, 1]
        dnn_predict = (torch.matmul(torch.max(torch.matmul(batchdataset, W1) + b1[:,None,:], torch.tensor([0.0]).to(self.device)), W2) + b2[:,None,:])
        dnn_predict_true = dnn_predict * std_y_train + mean_y_train # [B, n, 1]
        predict_mean = dnn_predict_true.mean(0)
        test_rmse = (((predict_mean - batchlabel)**2).mean())**(0.5)
        logpy_xz = -0.5 * (np.log(2*np.pi) - log_gamma[:,None,:]) - 0.5 * gamma_[:, None, :] * (dnn_predict_true - batchlabel[None, :, :])**2
        test_llk = (torch.logsumexp(logpy_xz.squeeze(2), dim=0).mean() - np.log(Z.shape[0]))
        return test_rmse.item(), test_llk.item()
    def predict_y(self, Z, batchdataset, mean_y_train, std_y_train, max_param = 50.0):
        """
        return the predicted response variable \hat{y} given the independent variables.
        """
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-1].reshape(-1,1) # [B, 1]
        dnn_predict = (torch.matmul(torch.max(torch.matmul(batchdataset, W1) + b1[:,None,:], torch.tensor([0.0]).to(self.device)), W2) + b2[:,None,:])
        dnn_predict_true = dnn_predict * std_y_train + mean_y_train
        return dnn_predict_true
    def model_selection(self, Z, batchdataset, batchlabel, mean_y_train, std_y_train, max_param = 50.0):
        """
        Adjust the heuristic loggamma if needed.
        """
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(-1, self.d, self.n_hidden) # [B, d, hidden]
        b1 = Z[:, (self.d) * self.n_hidden:(self.d+1) * self.n_hidden].reshape(-1, self.n_hidden) # [B, hidden]
        W2 = Z[:, (self.d+1) * self.n_hidden:(self.d+1) * self.n_hidden+self.n_hidden][:,:,None] # [B, hidden, 1]
        b2 = Z[:,-1].reshape(-1,1) # [B, 1]
        dnn_predict = (torch.matmul(torch.max(torch.matmul(batchdataset, W1) + b1[:,None,:], torch.tensor([0.0]).to(self.device)), W2) + b2[:,None,:])
        dnn_predict_true = dnn_predict * std_y_train + mean_y_train # [B, n, 1]
        log_gamma_heu = -torch.log(((dnn_predict_true - batchlabel[None, :, :])**2).mean(1))
        self.loggamma = log_gamma_heu

class GMM(torch.nn.Module):
    def __init__(self, dim, n_mixes, loc_scaling, log_var_scaling=0.1, seed=0,
                 n_test_set_samples=1000, use_gpu=True):
        super(GMM, self).__init__()
        self.seed = seed
        self.n_mixes = n_mixes
        self.dim = dim
        self.n_test_set_samples = n_test_set_samples

        mean = (torch.rand((n_mixes, dim)) - 0.5)*2 * loc_scaling
        log_var = torch.ones((n_mixes, dim)) * log_var_scaling

        self.register_buffer("cat_probs", torch.ones(n_mixes))
        self.register_buffer("locs", mean)
        self.register_buffer("scale_trils", torch.diag_embed(F.softplus(log_var)))

        self.device = "cuda" if use_gpu else "cpu"
        self.to(self.device)

    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()

    @property
    def distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs,
                                                     scale_tril=self.scale_trils,
                                                     validate_args=False)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                     component_distribution=com,
                                                     validate_args=False)

    def logp(self, x: torch.Tensor):
        log_prob = self.distribution.log_prob(x)
        # Very low probability samples can cause issues (we turn off validate_args of the
        # distribution object which typically raises an expection related to this.
        # We manually decrease the distributions log prob to prevent them having an effect on
        # the loss/buffer.
        #mask = torch.zeros_like(log_prob)
        #mask[log_prob < -1e4] = - torch.tensor(float("inf"))
        #log_prob = log_prob + mask
        return log_prob

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)


target_distribution = {
    "banana":Banana_shape,
    "multimodal":Multimodal,
    "x_shaped":X_shaped,
    "LRwaveform":LRwaveform,
    "Langevin_post": Langevin_post,
    "Bnn": Bnn,
    "gmm": GMM,
    "neal": Neal
}
