import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

class Encoder_overall(Module):

    """\
    Overall encoder.

    Parameters
    ----------
    dim_in_feat_omics1 : int
        Dimension of input features for omics1.
    dim_in_feat_omics2 : int
        Dimension of input features for omics2.
    dim_out_feat_omics1 : int
        Dimension of latent representation for omics1.
    dim_out_feat_omics2 : int
        Dimension of latent representation for omics2, which is the same as omics1.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    results: a dictionary including representations and modality weights.

    """

    def __init__(self, dim_input_list, dim_output):
        super(Encoder_overall, self).__init__()
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        self.MoE_list = nn.ModuleList()
        self.Modality_num = len(dim_input_list)
        for dim in dim_input_list:
            self.encoder_list.append(VAEEncoder(dim, dim_output))
            self.decoder_list.append(Decoder(dim_output, dim))
            self.MoE_list.append(MoE(128, 2))

        self.discriminator = MLP(dim_output * 2, dim_output, 2)

        self.PoE = ProductOfExperts()


    def forward(self, feat_list, adj_feature, adj_spatial, sampled_indices, zero_indices, train=True):
        v_emb_latent_list = []
        inv_emb_latent_list = []
        emb_latent_omics_list = []
        gates_list = []
        moe_loss_list = []
        mu_omics_list = []
        logvar_omics_list = []
        kl_list = []
        emb_recon_list = []


        for i in range(self.Modality_num):
            h_omics, mu_omics_v, logvar_omics_v, mu_omics_inv, logvar_omics_inv = self.encoder_list[i](
                feat_list[i], adj_feature)
            v_emb_latent_omics = self.reparameterize(mu_omics_v, logvar_omics_v)
            v_emb_latent_list.append(v_emb_latent_omics)
            inv_emb_latent_omics = self.reparameterize(mu_omics_inv, logvar_omics_inv)
            inv_emb_latent_list.append(inv_emb_latent_omics)
            gates_omics, moe_loss_omics = self.MoE_list[i](h_omics, train)
            gates_list.append(gates_omics)
            moe_loss_list.append(moe_loss_omics)
            mu_omics = mu_omics_v * gates_omics[:, 0].unsqueeze(1) + mu_omics_inv * gates_omics[:, 1].unsqueeze(1)
            logvar_omics = logvar_omics_v * gates_omics[:, 0].unsqueeze(1) + logvar_omics_inv * gates_omics[:,
                                                                                                    1].unsqueeze(1)
            mu_omics_list.append(mu_omics.unsqueeze(0))
            logvar_omics_list.append(logvar_omics.unsqueeze(0))
            emb_latent_omics = self.reparameterize(mu_omics, logvar_omics)
            emb_latent_omics_list.append(emb_latent_omics)
            kl_list.append(self.cal_kl_loss(mu_omics, logvar_omics))

        # MoE_Distribution

        mu_omics = torch.cat(mu_omics_list, dim=0)

        logvar_omics = torch.cat(logvar_omics_list, dim=0)

        self.inv_mu, self.inv_logvar = self.PoE(mu_omics, logvar_omics)

        emb_latent_combined = self.reparameterize(self.inv_mu, self.inv_logvar)

        discriminator_pred, mixed_labels = self.get_discriminator_pred(emb_latent_combined, sampled_indices, zero_indices)

        for i in range(self.Modality_num):
            emb_recon_omics = self.decoder_list[i](emb_latent_combined, adj_spatial)
            emb_recon_list.append(emb_recon_omics)



        results = {
                    'emb_latent_combined':emb_latent_combined,
                    'emb_recon_list':emb_recon_list,
                    'v_emb_latent_list': v_emb_latent_list,
                    'inv_emb_latent_list': inv_emb_latent_list,
                    'emb_latent_omics_list': emb_latent_omics_list,
                    'gates_list': gates_list,
                    'moe_loss_list': moe_loss_list,
                    'mu_omics_list': mu_omics_list,
                    'logvar_omics_list': logvar_omics_list,
                    'kl_div': sum(kl_list),
                    'moe_loss': sum(moe_loss_list),
                    'mixed_labels': mixed_labels,
                    'discriminator_pred': discriminator_pred
                   }

        return results

    def get_discriminator_pred(self, emb_latent_combined, one_indices, zero_indices):
        if one_indices is not None:
            one_paired_features = emb_latent_combined[one_indices]
            zero_paired_features = emb_latent_combined[zero_indices]

            M = one_paired_features.shape[0]
            indices = torch.randperm(M * 2)

            mixed_features = torch.cat((one_paired_features, zero_paired_features), dim=0)
            mixed_features = mixed_features[indices]

            mixed_labels = torch.cat((torch.ones(M), torch.zeros(M)))[indices]

            discriminator_pred = self.discriminator(mixed_features.view(mixed_features.shape[0], -1))

        else:
            mixed_labels = None
            N = emb_latent_combined.shape[0]

            emb_latent_combined_1 = emb_latent_combined.unsqueeze(1).expand(-1, N, -1)  # shape: (N, N, D)
            emb_latent_combined_2 = emb_latent_combined.unsqueeze(0).expand(N, -1, -1)  # shape: (N, N, D)
            print(emb_latent_combined_1.shape, emb_latent_combined_2.shape)
            mixed_features = torch.cat((emb_latent_combined_1, emb_latent_combined_2), dim=2)

            # Reshape to (N*N, 2*D) for batch processing
            mixed_features = mixed_features.view(-1, mixed_features.shape[-1])  # shape will be (N*N, 2*D)

            # Get predictions from the discriminator
            predictions = self.discriminator(mixed_features)

            # Apply softmax and reshape back to (N, N)
            discriminator_pred = F.softmax(predictions, dim=1)[:, 0].view(N, N)

        return discriminator_pred, mixed_labels

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def cal_kl_loss(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).mean().div(math.log(2))
        return kl

class Encoder_single(Module):

    """\
    Overall encoder.

    Parameters
    ----------
    dim_in_feat_omics1 : int
        Dimension of input features for omics1.
    dim_in_feat_omics2 : int
        Dimension of input features for omics2.
    dim_out_feat_omics1 : int
        Dimension of latent representation for omics1.
    dim_out_feat_omics2 : int
        Dimension of latent representation for omics2, which is the same as omics1.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    results: a dictionary including representations and modality weights.

    """

    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dropout=0.0, act=F.relu):
        super(Encoder_single, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act
        self.output_dim = dim_out_feat_omics1

        self.encoder_omics1 = VAEEncoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)

        self.discriminator = MLP(self.dim_out_feat_omics1 * 2, self.dim_out_feat_omics1, 2)

        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)

        self.PoE = ProductOfExperts()
        self.MoE_omics1 = MoE(128, 2)


    def forward(self, features_omics1, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2,
                one_indices=None, zero_indices=None, train=True):

        h_omics1, mu_omics1_v, logvar_omics1_v, mu_omics1_inv, logvar_omics1_inv = self.encoder_omics1(features_omics1, adj_feature_omics1)
        v_emb_latent_omics1 = self.reparameterize(mu_omics1_v, logvar_omics1_v)
        inv_emb_latent_omics1 = self.reparameterize(mu_omics1_inv, logvar_omics1_inv)

        # MoE_Distribution

        gates_omics1, moe_loss_omics1 = self.MoE_omics1(h_omics1, train)

        mu_omics1 = mu_omics1_v * gates_omics1[:, 0].unsqueeze(1) + mu_omics1_inv * gates_omics1[:, 1].unsqueeze(1)

        logvar_omics1 = logvar_omics1_v * gates_omics1[:, 0].unsqueeze(1) + logvar_omics1_inv * gates_omics1[:, 1].unsqueeze(1)

        self.inv_mu, self.inv_logvar = mu_omics1, logvar_omics1

        emb_latent_combined = self.reparameterize(self.inv_mu, self.inv_logvar)

        kl_combined = self.cal_kl_loss(mu_omics1_v, logvar_omics1_v)

        discriminator_pred, mixed_labels = self.get_discriminator_pred(emb_latent_combined, one_indices, zero_indices)


        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(inv_emb_latent_omics1, adj_spatial_omics2)



        results = {
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                   'emb_latent_omics1': v_emb_latent_omics1,
                   'inv_emb_latent_omics1': inv_emb_latent_omics1,
                   'v_emb_latent_omics1': v_emb_latent_omics1,
                   'mixed_labels': mixed_labels,
                   'discriminator_pred': discriminator_pred,
                   'kl_div': kl_combined,
                   'moe_loss': moe_loss_omics1,
                   }

        return results

    def get_discriminator_pred(self, emb_latent_combined, one_indices, zero_indices):

        if one_indices is not None:
            one_paired_features = emb_latent_combined[one_indices]
            zero_paired_features = emb_latent_combined[zero_indices]

            M = one_paired_features.shape[0]
            indices = torch.randperm(M * 2)

            mixed_features = torch.cat((one_paired_features, zero_paired_features), dim=0)
            mixed_features = mixed_features[indices]

            mixed_labels = torch.cat((torch.ones(M), torch.zeros(M)))[indices]

            discriminator_pred = self.discriminator(mixed_features.view(mixed_features.shape[0], -1))

        else:
            mixed_labels = None
            N = emb_latent_combined.shape[0]

            emb_latent_combined_1 = emb_latent_combined.unsqueeze(1).expand(-1, N, -1)  # shape: (N, N, D)
            emb_latent_combined_2 = emb_latent_combined.unsqueeze(0).expand(N, -1, -1)  # shape: (N, N, D)
            print(emb_latent_combined_1.shape, emb_latent_combined_2.shape)
            mixed_features = torch.cat((emb_latent_combined_1, emb_latent_combined_2), dim=2)

            # Reshape to (N*N, 2*D) for batch processing
            mixed_features = mixed_features.view(-1, mixed_features.shape[-1])  # shape will be (N*N, 2*D)

            # Get predictions from the discriminator
            predictions = self.discriminator(mixed_features)

            # Apply softmax and reshape back to (N, N)
            discriminator_pred = F.softmax(predictions, dim=1)[:, 0].view(N, N)

        return discriminator_pred, mixed_labels

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        return z

    def cal_kl_loss(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).mean().div(math.log(2))
        return kl

class Encoder(Module):

    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Latent representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        feat_embeding = torch.mm(feat, self.weight)
        x = torch.mm(adj, feat_embeding)

        return x

class Decoder(Module):

    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Reconstructed representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.in_feat))
        self.fc1 = nn.Linear(self.in_feat, self.out_feat)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.mm(adj, x)
        x = self.fc1(x)

        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5, if_bn=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.PReLU()
        # self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.if_bn = if_bn

        if self.if_bn:
            self.bn1 = nn.BatchNorm1d(hidden_size)

        self.reset_parameters(self.fc1.weight)
        self.reset_parameters(self.fc2.weight)

    def reset_parameters(self, weight):
        torch.nn.init.xavier_uniform_(weight)

    def forward(self, x):
        out = self.fc1(x)

        if self.if_bn:
            out = self.bn1(out)

        # out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        return out

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()
        return grad_output, None

class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)

class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        # self.dp = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.grl = GRL()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x, if_grl=True):
        # x = self.manual_instance_norm_2d(x)
        if if_grl:
            x = self.grl(x)
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dp(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        self.weight = Parameter(torch.FloatTensor(input_dim, 128))

        # self.GAT = GATLayer(input_dim, 128)

        # 假设我们用一个简单的全连接网络
        # self.fc1 = nn.Linear(128, 128)
        self.fc2_mu = nn.Linear(128, latent_dim)  # 均值
        self.fc2_logvar = nn.Linear(128, latent_dim)  # 对数方差

        self.fc2_mu_inv = nn.Linear(128, latent_dim)  # 均值
        self.fc2_logvar_inv = nn.Linear(128, latent_dim)  # 对数方差

        self.bn1 = nn.BatchNorm1d(128)

        self.reset_parameters(self.weight)
        # self.reset_parameters(self.GAT.weight)

        # self.reset_parameters(self.fc1.weight)
        self.reset_parameters(self.fc2_mu.weight)
        self.reset_parameters(self.fc2_logvar.weight)
        self.reset_parameters(self.fc2_mu_inv.weight)
        self.reset_parameters(self.fc2_logvar_inv.weight)

        # self.fc2_mu.bias.data.zero_()
        # self.fc2_logvar.bias.data.zero_()
        # self.fc2_mu_inv.bias.data.zero_()
        # self.fc2_logvar_inv.bias.data.zero_()

    def reset_parameters(self, weight):
        torch.nn.init.xavier_uniform_(weight)

    def manual_instance_norm_2d(self, x, eps=1e-5):
        # x shape: (N, D)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)  
        x_norm = (x - mean) / torch.sqrt(var + eps)
        return x_norm

    def forward(self, x, adj):
        feat_embeding = torch.mm(x, self.weight)
        x = torch.mm(adj, feat_embeding)

        # x = self.GAT(x, adj)

        x = self.manual_instance_norm_2d(x)

        h = self.bn1(x)

        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)

        mu_inv = self.fc2_mu_inv(h)
        logvar_inv = self.fc2_logvar_inv(h)
        return h, mu, logvar, mu_inv, logvar_inv

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        # print(mu.shape, pd_mu.shape)
        pd_var = 1. / torch.sum(T, dim=0) * 1
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar

class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, num_experts, noisy_gating=True, k=2, coef=1e-2, num_experts_1hop=None):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        # self.output_size = output_size
        self.input_size = input_size
        self.k = k
        self.loss_coef = coef
        if not num_experts_1hop:
            self.num_experts_1hop = num_experts # by default, all experts are hop-1 experts.
        else:
            assert num_experts_1hop <= num_experts
            self.num_experts_1hop = num_experts_1hop
        # instantiate experts
        # self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        # self.experts_conv = experts_conv
        # self.experts_bn = experts_bn

        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        # torch.nn.init.xavier_uniform_(self.w_gate)
        # torch.nn.init.xavier_uniform_(self.w_noise)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, training=True):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss_cv_squared = self.cv_squared(importance) + self.cv_squared(load)


        return gates, loss_cv_squared




