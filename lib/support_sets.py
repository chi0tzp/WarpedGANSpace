import torch
from torch import nn


class SupportSets(nn.Module):
    """Support Sets class

        TODO: K = as many as the desired interpretable paths
            Each support set contains `num_support_dipoles` support vector dipoles -- i.e., "antipodal" support vectors
            with opposite weights alpha (-1, +1) and the same gamma (scale) parameter. During training the position of
            support vectors are being optimized, while weights alpha and scale parameters gamma are
    """
    def __init__(self, num_support_sets, num_support_dipoles, support_vectors_dim,
                 learn_alphas=False, learn_gammas=False, gamma=None):
        """ SupportSets constructor.

        Args:
            num_support_sets (int)    : number of support sets (each one defining a warping function)
            num_support_dipoles (int) : number of support dipoles per support set (per warping function)
            support_vectors_dim (int) : dimensionality of support vectors (latent space dimensionality, z_dim)
            learn_alphas (bool)       : learn RBF alphas
            learn_gammas (bool)       : learn RBF gammas
            gamma (float)             : RBF gamma parameter (by default set to the inverse of the latent space
                                        dimensionality)
        """
        super(SupportSets, self).__init__()
        self.num_support_sets = num_support_sets
        self.num_support_dipoles = num_support_dipoles
        self.support_vectors_dim = support_vectors_dim
        self.learn_alphas = learn_alphas
        self.learn_gammas = learn_gammas
        self.gamma = gamma
        self.loggamma = torch.log(torch.scalar_tensor(self.gamma))

        # TODO: add comment
        self.r = 3.0
        self.r_min = 1.0
        self.r_max = 4.0
        self.r_mean = 0.5 * (self.r_min + self.r_max)
        self.radii = torch.arange(self.r_min, self.r_max, (self.r_max - self.r_min)/self.num_support_sets)

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
        # ************************************************************************************************************ #

        ################################################################################################################
        ##                                                                                                            ##
        ##                                            [ ALPHAS: (K, N) ]                                              ##
        ##                                                                                                            ##
        ################################################################################################################
        # REVIEW: Define alphas as parameters (learnable or non-learnable)
        self.ALPHAS = nn.Parameter(data=torch.zeros(self.num_support_sets, 2 * self.num_support_dipoles),
                                   requires_grad=self.learn_alphas)

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
        # Define RBF gammas
        self.LOGGAMMA = nn.Parameter(data=self.loggamma * torch.ones(self.num_support_sets, 1),
                                     requires_grad=self.learn_gammas)

    def forward(self, support_sets_mask, z):
        """TODO: +++

        Args:
            support_sets_mask (torch.Tensor): TODO: +++ -- size: +++
            z (torch.Tensor): input latent codes -- size: TODO: +++

        Returns:
            Normalized grad of f evaluated at given z -- size: (bs, dim).

        """
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

        # REVIEW: Fix `grad_f` formula
        # grad_f = -2 * (alphas_batch * torch.exp(-gammas_batch * (torch.norm(D, dim=2) ** 2).unsqueeze(dim=2)) * D).sum(dim=1)
        grad_f = -2 * (alphas_batch * gammas_batch * torch.exp(-gammas_batch * (torch.norm(D, dim=2) ** 2).unsqueeze(dim=2)) * D).sum(dim=1)
        
        # Return normalized grad of f at z
        return grad_f / torch.norm(grad_f, dim=1, keepdim=True)
