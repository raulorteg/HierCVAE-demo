import numpy as np
import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from hgraph.decoder import CondHierMPNDecoder, HierMPNDecoder
from hgraph.encoder import HierMPNEncoder
from hgraph.mol_graph import MolGraph
from hgraph.nnutils import *


def make_cuda(tensors, device: str):
    """
    Function used to unpack the tree and graph tensors and send them to the correct device in use
    (cuda if possible, else cpu)
    :param tensors: tuple containing the tree and graph tensors
    :device: string denoting the device to send them to ("cuda" if available, else "cpu")
    :return: tree and grpah tensors unpacked and sent to the correct device in use.
    """
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    # tree_tensors = [make_tensor(x).cuda().long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    # graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    tree_tensors = [make_tensor(x).to(device).long() for x in tree_tensors[:-1]] + [
        tree_tensors[-1]
    ]
    graph_tensors = [make_tensor(x).to(device).long() for x in graph_tensors[:-1]] + [
        graph_tensors[-1]
    ]

    return tree_tensors, graph_tensors


class HierVAE(nn.Module):
    """
    Hierarchical Variational Autoencoder (HierVAE).
    """

    def __init__(self, args):
        super(HierVAE, self).__init__()

        # define the device so the model can work also on cpu if cuda not available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = HierMPNEncoder(
            args.vocab,
            args.atom_vocab,
            args.rnn_type,
            args.embed_size,
            args.hidden_size,
            args.depthT,
            args.depthG,
            args.dropout,
        )
        self.decoder = HierMPNDecoder(
            args.vocab,
            args.atom_vocab,
            args.rnn_type,
            args.embed_size,
            args.hidden_size,
            args.latent_size,
            args.diterT,
            args.diterG,
            args.dropout,
        )
        self.encoder.tie_embedding(self.decoder.hmpn)
        self.latent_size = args.latent_size

        self.R_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.R_var = nn.Linear(args.hidden_size, args.latent_size)

    def rsample(self, z_vecs, W_mean, W_var, perturb: bool = True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))
        kl_loss = (
            -0.5
            * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var))
            / batch_size
        )
        # epsilon = torch.randn_like(z_mean).cuda()
        epsilon = torch.randn_like(z_mean).to(self.device)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def sample(self, batch_size: int, greedy):
        """
        Method for sampling randomly from a normal gaussian distribution the latent space and decode
        :param batch_size: int number of samples per batch
        :param greedy: ¿?boolean flag ...
        :return: decoded samples into molecules
        """
        # root_vecs = torch.randn(batch_size, self.latent_size).cuda()
        root_vecs = torch.randn(batch_size, self.latent_size).to(self.device)
        return self.decoder.decode(
            (root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150
        )

    def sample_axis(
        self, batch_size: int, greedy, axis: int, min_: float = -1.5, max_: float = 1.5
    ):
        """
        Method for sampling points in the latent space uniformly while moving alongside a specified dimension/axis
        of the latent space. (e.g if axis=1 then samples from the latent space are [0,-1.5,0,....0], [0,-1.2,0,....0], ...., [0,+1.5,0,....0],
        that is we move through the mean of the latent space while varying one dimension uniformly at a time) This is used to try an infer what each
        dimension/axis in the latent space is encoding.

        :param batch_size: int number of samples per batch
        :param greedy: ¿?boolean flag ...
        :param axis: int index of the axis to move along.
        :param min_: float value minimum value to sample (defaut -1.5)
        :param max_: float value maximum value to sample (default= +1.5)
        :return: decoded samples into molecules
        """
        root_vecs = torch.zeros(batch_size, self.latent_size)
        axis_value = np.linspace(min_, max_, batch_size)
        for i in range(batch_size):
            root_vecs[i, axis] = axis_value[i]
        root_vecs = root_vecs.cuda().to(self.device)
        return self.decoder.decode(
            (root_vecs, root_vecs[:, :-1], root_vecs[:, :-1]),
            greedy=greedy,
            max_decode_step=150,
        )

    def reconstruct(self, batch):
        graphs, tensors, _ = batch
        tree_tensors, graph_tensors = tensors = make_cuda(tensors, device=self.device)
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

        root_vecs, root_kl = self.rsample(
            root_vecs, self.R_mean, self.R_var, perturb=False
        )
        return self.decoder.decode(
            (root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150
        )

    def forward(self, graphs, tensors, orders, beta, perturb_z=True):
        tree_tensors, graph_tensors = tensors = make_cuda(tensors, device=self.device)

        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb_z)
        kl_div = root_kl

        loss, wacc, iacc, tacc, sacc = self.decoder(
            (root_vecs, root_vecs, root_vecs), graphs, tensors, orders
        )
        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc


class CondHierVAE(nn.Module):
    """
    Conditional Hierarchical Variational Autoencoder (CondHierVAE).
    MOdified version of the HierVAE class to allow for conditonal generation of molecules
    based on a target property.
    """

    def __init__(self, args):
        super(CondHierVAE, self).__init__()

        # define the device so the model can work also on cpu if cuda not available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = HierMPNEncoder(
            args.vocab,
            args.atom_vocab,
            args.rnn_type,
            args.embed_size,
            args.hidden_size,
            args.depthT,
            args.depthG,
            args.dropout,
        )
        self.decoder = CondHierMPNDecoder(
            args.vocab,
            args.atom_vocab,
            args.rnn_type,
            args.embed_size,
            args.hidden_size,
            args.latent_size,
            args.diterT,
            args.diterG,
            args.dropout,
        )
        self.encoder.tie_embedding(self.decoder.hmpn)
        self.latent_size = args.latent_size

        self.R_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.R_var = nn.Linear(args.hidden_size, args.latent_size)

    def rsample(self, z_vecs, W_mean, W_var, perturb: bool = True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))
        kl_loss = (
            -0.5
            * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var))
            / batch_size
        )
        # epsilon = torch.randn_like(z_mean).cuda()
        epsilon = torch.randn_like(z_mean).to(self.device)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def sample(self, batch_size: int, greedy):
        """
        Method for sampling randomly from a normal gaussian distribution the latent space and decode
        :param batch_size: int number of samples per batch
        :param greedy: ¿?boolean flag ...
        :return: decoded samples into molecules
        """
        root_vecs = torch.randn(batch_size, self.latent_size).to(self.device)
        return self.decoder.decode(
            (root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150
        )

    def sample_axis(
        self, batch_size: int, greedy, axis: int, min_: float = -1.5, max_: float = 1.5
    ):
        """
        Method for sampling points in the latent space uniformly while moving alongside a specified dimension/axis
        of the latent space. (e.g if axis=1 then samples from the latent space are [0,-1.5,0,....0], [0,-1.2,0,....0], ...., [0,+1.5,0,....0],
        that is we move through the mean of the latent space while varying one dimension uniformly at a time) This is used to try an infer what each
        dimension/axis in the latent space is encoding.

        :param batch_size: int number of samples per batch
        :param greedy: ¿?boolean flag ...
        :param axis: int index of the axis to move along.
        :param min_: float value minimum value to sample (defaut -1.5)
        :param max_: float value maximum value to sample (default= +1.5)
        :return: decoded samples into molecules
        """
        root_vecs = torch.zeros(batch_size, self.latent_size + 1)
        axis_value = np.linspace(min_, max_, batch_size)
        for i in range(batch_size):
            root_vecs[i, axis] = axis_value[i]
        root_vecs = root_vecs.to(self.device)
        return self.decoder.decode(
            (root_vecs, root_vecs[:, :-1], root_vecs[:, :-1]),
            greedy=greedy,
            max_decode_step=150,
        )

    def sample_cond(self, batch_size: int, greedy, cond: float):
        """
        Method for sampling randomly from a normal gaussian distribution the latent space with a condition and decode conditionally
        on a variable cond (property targetted)
        :param batch_size: int number of samples per batch
        :param greedy: ¿?boolean flag ...
        :param cond: float value of the targetted condition that the conditional generation depends on
        :return: decoded samples into molecules
        """
        root_vecs = torch.randn(batch_size, self.latent_size + 1)
        for i in range(batch_size):
            root_vecs[i, -1] = cond
        root_vecs = root_vecs.to(self.device)
        return self.decoder.decode(
            (root_vecs, root_vecs[:, :-1], root_vecs[:, :-1]),
            greedy=greedy,
            max_decode_step=150,
        )

    def reconstruct(self, batch):
        graphs, tensors, _ = batch
        tree_tensors, graph_tensors = tensors = make_cuda(tensors, device=self.device)
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

        root_vecs, root_kl = self.rsample(
            root_vecs, self.R_mean, self.R_var, perturb=False
        )
        return self.decoder.decode(
            (root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150
        )

    def forward(
        self, graphs, tensors, orders, cond: list, beta, perturb_z: bool = True
    ):
        tree_tensors, graph_tensors = tensors = make_cuda(tensors, device=self.device)

        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb_z)
        kl_div = root_kl

        loss, wacc, iacc, tacc, sacc = self.decoder(
            (root_vecs, root_vecs, root_vecs), graphs, tensors, orders, cond
        )
        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc
