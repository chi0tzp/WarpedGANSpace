import torch
from torch import nn


class SupportSets(nn.Module):
    def __init__(self, num_support_sets, num_support_dipoles, support_vectors_dim, expected_latent_norm=20.0,
                 learn_alphas=False, learn_gammas=False, gamma=None):
        """SupportSets class constructor.

        Args:
            num_support_sets (int)    : number of support sets (each one defining a warping function)
            num_support_dipoles (int) : number of support dipoles per support set (per warping function)
            support_vectors_dim (int) : dimensionality of support vectors (latent space dimensionality)
            expected_latent_norm      : expected norm of the latent codes for the given GAN type
            learn_alphas (bool)       : learn RBF alphas
            learn_gammas (bool)       : learn RBF gammas
            gamma (float)             : RBF gamma parameter
        """
        super(SupportSets, self).__init__()
        self.num_support_sets = num_support_sets
        self.num_support_dipoles = num_support_dipoles
        self.support_vectors_dim = support_vectors_dim
        self.expected_latent_norm = expected_latent_norm
        self.learn_alphas = learn_alphas
        self.learn_gammas = learn_gammas
        self.gamma = gamma
        self.loggamma = torch.log(torch.scalar_tensor(self.gamma))

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
        # Initialisation of support sets -- Choose r_min and r_max based on the expected latent norm; i.e., the expected
        # norm of a latent code drawn from the latent space (Z, W or W+) for the given truncation parameter
        self.r_min = 0.8 * self.expected_latent_norm
        self.r_max = 0.9 * self.expected_latent_norm

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
        self.LOGGAMMA = nn.Parameter(data=self.loggamma * torch.ones(self.num_support_sets, 1),
                                     requires_grad=self.learn_gammas)

    def forward(self, support_sets_mask, z):
        # Get RBF support sets batch
        support_sets_batch = torch.matmul(support_sets_mask, self.SUPPORT_SETS)
        support_sets_batch = support_sets_batch.reshape(-1, 2 * self.num_support_dipoles, self.support_vectors_dim)

        # Get batch of RBF alpha parameters
        alphas_batch = torch.matmul(support_sets_mask, self.ALPHAS).unsqueeze(dim=2)

        # Get batch of RBF gamma/log(gamma) parameters
        if self.learn_gammas:
            gammas_batch = torch.exp(torch.matmul(support_sets_mask, self.LOGGAMMA).unsqueeze(dim=2))
        else:
            gammas_batch = self.gamma * torch.ones(z.size()[0], 2 * self.num_support_dipoles, 1)

        # Calculate grad of f at z
        D = z.unsqueeze(dim=1).repeat(1, 2 * self.num_support_dipoles, 1) - support_sets_batch
        grad_f = -2 * (alphas_batch * gammas_batch *
                       torch.exp(-gammas_batch * (torch.norm(D, dim=2) ** 2).unsqueeze(dim=2)) * D).sum(dim=1)

        # Return normalized grad of f at z
        return grad_f / torch.norm(grad_f, dim=1, keepdim=True)
