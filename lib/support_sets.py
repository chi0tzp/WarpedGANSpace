import sys
import numpy as np
import torch
from torch import nn


class SupportSets(nn.Module):
    def __init__(self, num_support_sets, num_support_dipoles, support_vectors_dim, jung_radius,
                 learn_alphas=False, learn_gammas=False, beta=0.001):
        """SupportSets class constructor.

        Args:
            num_support_sets (int)    : number of support sets (each one defining a warping function)
            num_support_dipoles (int) : number of support dipoles per support set (per warping function)
            support_vectors_dim (int) : dimensionality of support vectors (latent space dimensionality)
            jung_radius               : TODO: add comment
            learn_alphas (bool)       : learn RBF alphas
            learn_gammas (bool)       : learn RBF gammas
            beta (float)              : RBF beta parameter
        """
        super(SupportSets, self).__init__()
        self.num_support_sets = num_support_sets
        self.num_support_dipoles = num_support_dipoles
        self.support_vectors_dim = support_vectors_dim
        self.jung_radius = jung_radius
        self.learn_alphas = learn_alphas
        self.learn_gammas = learn_gammas
        self.beta = beta

        ################################################################################################################
        ##                                                                                                            ##
        ##                                        [ SUPPORT_SETS: (K, N, d) ]                                         ##
        ##                                                                                                            ##
        ################################################################################################################
        # Define learnable parameters ofr RBF support sets:
        #   K sets of N pairs of d-dimensional (antipodal) support vector sets
        self.SUPPORT_SETS = nn.Parameter(data=torch.ones(self.num_support_sets,
                                                         2 * self.num_support_dipoles * self.support_vectors_dim),
                                         requires_grad=True)
        # Initialisation of support sets -- Choose r_min and r_max based on TODO: +++

        self.r_min = 0.9 * self.jung_radius
        self.r_max = 1.25 * self.jung_radius

        # self.r_min = 1
        # self.r_max = 4

        self.radii = torch.arange(self.r_min, self.r_max, (self.r_max - self.r_min) / self.num_support_sets)
        SUPPORT_SETS = torch.zeros(self.num_support_sets, 2 * self.num_support_dipoles, self.support_vectors_dim)
        for k in range(self.num_support_sets):
            SV_set = []
            for i in range(self.num_support_dipoles):
                SV = torch.randn(1, self.support_vectors_dim)
                SV_set.extend([SV, -SV])
            SV_set = torch.cat(SV_set)
            SV_set = self.radii[k] * SV_set / torch.norm(SV_set, dim=1, keepdim=True)
            SUPPORT_SETS[k, :] = SV_set

        # Reshape support sets tensor into a matrix and initialize support sets matrix
        self.SUPPORT_SETS.data = SUPPORT_SETS.reshape(self.num_support_sets,
                                                      2 * self.num_support_dipoles * self.support_vectors_dim).clone()

        ################################################################################################################
        ##                                                                                                            ##
        ##                                            [ ALPHAS: (K, N) ]                                              ##
        ##                                                                                                            ##
        ################################################################################################################
        # Define alphas as parameters (learnable or non-learnable)
        self.ALPHAS = nn.Parameter(data=torch.zeros(self.num_support_sets, 2 * self.num_support_dipoles),
                                   requires_grad=self.learn_alphas)
        # Initialize alphas
        for k in range(self.num_support_sets):
            a = []
            for _ in range(self.num_support_dipoles):
                a.extend([1, -1])
            self.ALPHAS.data[k] = torch.Tensor(a)

        ################################################################################################################
        ##                                                                                                            ##
        ##                                            [ GAMMAS: (K, N) ]                                              ##
        ##                                                                                                            ##
        ################################################################################################################
        # Define RBF gammas and initialize

        # self.LOGGAMMA = nn.Parameter(data=torch.ones(self.num_support_sets, 1), requires_grad=self.learn_gammas)
        # for k in range(self.num_support_sets):
        #     g = -np.log(self.beta) / ((2 * self.radii[k]) ** 2)
        #     self.LOGGAMMA.data[k] = torch.log(torch.Tensor([g]))

        self.LOGGAMMA = nn.Parameter(
            data=torch.log(torch.scalar_tensor(1.0 / self.support_vectors_dim)) * torch.ones(self.num_support_sets, 1),
            requires_grad=self.learn_gammas)

    def forward(self, support_sets_mask, z):
        # Get RBF support sets batch
        support_sets_batch = torch.matmul(support_sets_mask, self.SUPPORT_SETS)
        support_sets_batch = support_sets_batch.reshape(-1, 2 * self.num_support_dipoles, self.support_vectors_dim)

        # Get batch of RBF alpha parameters
        alphas_batch = torch.matmul(support_sets_mask, self.ALPHAS).unsqueeze(dim=2)

        # Get batch of RBF gamma/log(gamma) parameters
        gammas_batch = torch.exp(torch.matmul(support_sets_mask, self.LOGGAMMA).unsqueeze(dim=2))

        # Calculate grad of f at z
        D = z.unsqueeze(dim=1).repeat(1, 2 * self.num_support_dipoles, 1) - support_sets_batch
        grad_f = -2 * (alphas_batch * gammas_batch *
                       torch.exp(-gammas_batch * (torch.norm(D, dim=2) ** 2).unsqueeze(dim=2)) * D).sum(dim=1)

        # Return normalized grad of f at z
        return grad_f / torch.norm(grad_f, dim=1, keepdim=True)
