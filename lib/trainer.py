import sys
import os
import os.path as osp
import json
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import time
import shutil
from .config import STYLEGAN_LAYERS
from .aux import TrainingStatTracker, update_progress, update_stdout, sec2dhms


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Trainer(object):
    def __init__(self, params=None, exp_dir=None, use_cuda=False, multi_gpu=False):
        if params is None:
            raise ValueError("Cannot build a Trainer instance with empty params: params={}".format(params))
        else:
            self.params = params
        self.use_cuda = use_cuda
        self.multi_gpu = multi_gpu

        # Set output directory for current experiment (wip)
        self.wip_dir = osp.join("experiments", "wip", exp_dir)

        # Set directory for completed experiment
        self.complete_dir = osp.join("experiments", "complete", exp_dir)

        # Create log subdirectory and define stat.json file
        self.stats_json = osp.join(self.wip_dir, 'stats.json')
        if not osp.isfile(self.stats_json):
            with open(self.stats_json, 'w') as out:
                json.dump({}, out)

        # Create models sub-directory
        self.models_dir = osp.join(self.wip_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        # Define checkpoint model file
        self.checkpoint = osp.join(self.models_dir, 'checkpoint.pt')

        # Define cross entropy loss function
        self.cross_entropy = nn.CrossEntropyLoss()

        # Array of iteration times
        self.iter_times = np.array([])

        # Set up training statistics tracker
        self.stat_tracker = TrainingStatTracker()

    def get_starting_iteration(self, support_sets, reconstructor):
        """Check if checkpoint file exists (under `self.models_dir`) and set starting iteration at the checkpoint
        iteration; also load checkpoint weights to `support_sets` and `reconstructor`. Otherwise, set starting
        iteration to 1 in order to train from scratch.

        Returns:
            starting_iter (int): starting iteration

        """
        starting_iter = 1
        if osp.isfile(self.checkpoint):
            checkpoint_dict = torch.load(self.checkpoint)
            starting_iter = checkpoint_dict['iter']
            support_sets.load_state_dict(checkpoint_dict['support_sets'])
            reconstructor.load_state_dict(checkpoint_dict['reconstructor'])
        return starting_iter

    def log_progress(self, iteration, mean_iter_time, elapsed_time, eta):
        """Log progress in terms of batch accuracy, classification and regression losses and ETA.

        Args:
            iteration (int)        : current iteration
            mean_iter_time (float) : mean iteration time
            elapsed_time (float)   : elapsed time until current iteration
            eta (float)            : estimated time of experiment completion

        """
        # Get current training stats (for the previous `self.params.log_freq` steps) and flush them
        stats = self.stat_tracker.get_means()

        # Update training statistics json file
        with open(self.stats_json) as f:
            stats_dict = json.load(f)
        stats_dict.update({iteration: stats})
        with open(self.stats_json, 'w') as out:
            json.dump(stats_dict, out)

        # Flush training statistics tracker
        self.stat_tracker.flush()

        update_progress("  \\__.Training [bs: {}] [iter: {:06d}/{:06d}] ".format(
            self.params.batch_size, iteration, self.params.max_iter), self.params.max_iter, iteration + 1)
        if iteration < self.params.max_iter - 1:
            print()
        print("      \\__Batch accuracy      : {:.03f}".format(stats['accuracy']))
        print("      \\__Classification loss : {:.08f}".format(stats['classification_loss']))
        print("      \\__Regression loss     : {:.08f}".format(stats['regression_loss']))
        print("      \\__Total loss          : {:.08f}".format(stats['total_loss']))
        print("         ===================================================================")
        print("      \\__Mean iter time      : {:.3f} sec".format(mean_iter_time))
        print("      \\__Elapsed time        : {}".format(sec2dhms(elapsed_time)))
        print("      \\__ETA                 : {}".format(sec2dhms(eta)))
        print("         ===================================================================")
        update_stdout(10)

    def train(self, generator, support_sets, reconstructor):
        """Training function.

        Args:
            generator     : non-trainable (pre-trained) GAN generator
            support_sets  : trainable latent paths model
            reconstructor : trainable reconstructor

        """
        # Save initial `support_sets` model as `support_sets_init.pt`
        torch.save(support_sets.state_dict(), osp.join(self.models_dir, 'support_sets_init.pt'))

        # Set `generator` to evaluation mode, `support_sets` and `reconstructor` to training mode, and upload
        # models to GPU if `self.use_cuda` is set (i.e., if args.cuda and torch.cuda.is_available is True).
        if self.use_cuda:
            generator.cuda().eval()
            support_sets.cuda().train()
            reconstructor.cuda().train()
        else:
            generator.eval()
            support_sets.train()
            reconstructor.train()

        # Set support sets optimizer
        support_sets_optim = torch.optim.Adam(support_sets.parameters(), lr=self.params.support_set_lr)

        # Set shift predictor optimizer
        reconstructor_optim = torch.optim.Adam(reconstructor.parameters(), lr=self.params.reconstructor_lr)

        # Get starting iteration
        starting_iter = self.get_starting_iteration(support_sets, reconstructor)

        # Parallelize `generator` and `reconstructor` into multiple GPUs, if available and `multi_gpu=True`.
        if self.multi_gpu:
            print("#. Parallelize G, R over {} GPUs...".format(torch.cuda.device_count()))
            generator = DataParallelPassthrough(generator)
            reconstructor = DataParallelPassthrough(reconstructor)
            cudnn.benchmark = True

        # Check starting iteration
        if starting_iter == self.params.max_iter:
            print("#. This experiment has already been completed and can be found @ {}".format(self.wip_dir))
            print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
            try:
                shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
                print("  \\__Done!")
            except IOError as e:
                print("  \\__Already exists -- {}".format(e))
            sys.exit()
        print("#. Start training from iteration {}".format(starting_iter))

        # Get experiment's start time
        t0 = time.time()

        # Start training
        for iteration in range(starting_iter, self.params.max_iter + 1):

            # Get current iteration's start time
            iter_t0 = time.time()

            # Set gradients to zero
            generator.zero_grad()
            support_sets.zero_grad()
            reconstructor.zero_grad()

            # Sample latent codes from standard Gaussian
            z = torch.randn(self.params.batch_size, generator.dim_z)
            if self.use_cuda:
                z = z.cuda()

            # Generate images for the given latent codes
            latent_code = z
            if 'stylegan' in self.params.gan:
                if self.params.stylegan_space == 'W':
                    latent_code = generator.get_w(z, truncation=self.params.truncation)[:, 0, :]
                elif self.params.stylegan_space == 'W+':
                    latent_code = generator.get_w(z, truncation=self.params.truncation)
            img = generator(latent_code)

            # Sample indices of shift vectors (`self.params.batch_size` out of `self.params.num_support_sets`)
            target_support_sets_indices = torch.randint(0, self.params.num_support_sets, [self.params.batch_size])
            if self.use_cuda:
                target_support_sets_indices = target_support_sets_indices.cuda()

            # Sample shift magnitudes from uniform distributions
            #   U[self.params.min_shift_magnitude, self.params.max_shift_magnitude], and
            #   U[-self.params.max_shift_magnitude, self.params.min_shift_magnitude]
            # Create a pool of shift magnitudes of 2 * `self.params.batch_size` shifts (half negative, half positive)
            # and sample `self.params.batch_size` of them
            shift_magnitudes_pos = (self.params.min_shift_magnitude - self.params.max_shift_magnitude) * \
                torch.rand(target_support_sets_indices.size()) + self.params.max_shift_magnitude
            shift_magnitudes_neg = (self.params.min_shift_magnitude - self.params.max_shift_magnitude) * \
                torch.rand(target_support_sets_indices.size()) - self.params.min_shift_magnitude
            shift_magnitudes_pool = torch.cat((shift_magnitudes_neg, shift_magnitudes_pos))

            shift_magnitudes_ids = torch.arange(len(shift_magnitudes_pool), dtype=torch.float)
            target_shift_magnitudes = shift_magnitudes_pool[torch.multinomial(input=shift_magnitudes_ids,
                                                                              num_samples=self.params.batch_size,
                                                                              replacement=False)]
            if self.use_cuda:
                target_shift_magnitudes = target_shift_magnitudes.cuda()

            # Create support sets mask of size (batch_size, num_support_sets) in the form:
            #       support_sets_mask[i] = [0, ..., 0, 1, 0, ..., 0]
            support_sets_mask = torch.zeros([self.params.batch_size, self.params.num_support_sets])
            if self.use_cuda:
                support_sets_mask = support_sets_mask.cuda()
            for i, (index, val) in enumerate(zip(target_support_sets_indices, target_shift_magnitudes)):
                support_sets_mask[i][index] += 1.0

            # Calculate shift vectors for the given latent codes -- in the case of StyleGAN, shifts live in the
            # self.params.stylegan_space, i.e., in Z-, W-, or W+-space. In the Z-/W-space the dimensionality of the
            # latent space is 512. In the case of W+-space, the dimensionality is 512 * (self.params.stylegan_layer + 1)
            if ('stylegan' in self.params.gan) and (self.params.stylegan_space == 'W+'):
                shift = target_shift_magnitudes.reshape(-1, 1) * support_sets(
                    support_sets_mask, latent_code[:, :self.params.stylegan_layer + 1, :].reshape(latent_code.shape[0],
                                                                                                  -1))
            else:
                shift = target_shift_magnitudes.reshape(-1, 1) * support_sets(support_sets_mask, latent_code)

            # Generate images the shifted latent codes
            if ('stylegan' in self.params.gan) and (self.params.stylegan_space == 'W+'):
                latent_code_reshaped = latent_code.reshape(latent_code.shape[0], -1)
                shift = F.pad(input=shift,
                              pad=(0, (STYLEGAN_LAYERS[self.params.gan] - 1 - self.params.stylegan_layer) * 512),
                              mode='constant',
                              value=0)
                latent_code_shifted = latent_code_reshaped + shift
                latent_code_shifted_reshaped = latent_code_shifted.reshape_as(latent_code)
                img_shifted = generator(latent_code_shifted_reshaped)
            else:
                img_shifted = generator(latent_code + shift)

            # Predict support sets indices and shift magnitudes
            predicted_support_sets_indices, predicted_shift_magnitudes = reconstructor(img, img_shifted)

            # Calculate classification (cross entropy) and regression (mean absolute error) losses
            classification_loss = self.cross_entropy(predicted_support_sets_indices, target_support_sets_indices)
            regression_loss = torch.mean(torch.abs(predicted_shift_magnitudes - target_shift_magnitudes))

            # Calculate total loss and back-propagate
            loss = self.params.lambda_cls * classification_loss + self.params.lambda_reg * regression_loss
            loss.backward()

            # Perform optimization step (parameter update)
            support_sets_optim.step()
            reconstructor_optim.step()

            # Update statistics tracker
            self.stat_tracker.update(accuracy=torch.mean((torch.argmax(predicted_support_sets_indices, dim=1) ==
                                                          target_support_sets_indices).to(torch.float32)).detach(),
                                     classification_loss=classification_loss.item(),
                                     regression_loss=regression_loss.item(),
                                     total_loss=loss.item())

            # Get time of completion of current iteration
            iter_t = time.time()

            # Compute elapsed time for current iteration and append to `iter_times`
            self.iter_times = np.append(self.iter_times, iter_t - iter_t0)

            # Compute elapsed time so far
            elapsed_time = iter_t - t0

            # Compute rolling mean iteration time
            mean_iter_time = self.iter_times.mean()

            # Compute estimated time of experiment completion
            eta = elapsed_time * ((self.params.max_iter - iteration) / (iteration - starting_iter + 1))

            # Log progress in stdout
            if iteration % self.params.log_freq == 0:
                self.log_progress(iteration, mean_iter_time, elapsed_time, eta)

            # Save checkpoint model file and support_sets / reconstructor model state dicts after current iteration
            if iteration % self.params.ckp_freq == 0:
                # Build checkpoint dict
                checkpoint_dict = {
                    'iter': iteration,
                    'support_sets': support_sets.state_dict(),
                    'reconstructor': reconstructor.module.state_dict() if self.multi_gpu else reconstructor.state_dict()
                }
                torch.save(checkpoint_dict, self.checkpoint)
        # === End of training loop ===

        # Get experiment's total elapsed time
        elapsed_time = time.time() - t0

        # Save final support sets model
        support_sets_model_filename = osp.join(self.models_dir, 'support_sets.pt')
        torch.save(support_sets.state_dict(), support_sets_model_filename)

        # Save final shift predictor model
        reconstructor_model_filename = osp.join(self.models_dir, 'reconstructor.pt')
        torch.save(reconstructor.module.state_dict() if self.multi_gpu else reconstructor.state_dict(),
                   reconstructor_model_filename)

        for _ in range(10):
            print()
        print("#.Training completed -- Total elapsed time: {}.".format(sec2dhms(elapsed_time)))

        print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
        try:
            shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
            print("  \\__Done!")
        except IOError as e:
            print("  \\__Already exists -- {}".format(e))
