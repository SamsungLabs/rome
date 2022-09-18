import torch
from torch import nn
import apex
from apex import amp

import argparse
import os
import pathlib
import importlib
import ssl
import sys
from tqdm import tqdm
import functools

from src.utils import args as args_utils
from src.utils.logger import Logger


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        # Initialize and apply general options
        ssl._create_default_https_context = ssl._create_unverified_context
        torch.manual_seed(args.random_seed)
        if args.num_gpus > 0:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.random_seed)

        if args.torch_home:
            os.environ['TORCH_HOME'] = args.torch_home

        self.args = args

        # Set distributed training options
        if args.num_gpus <= 1:
            self.rank = 0

        elif args.num_gpus > 1 and args.num_gpus <= 8:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.rank)

        elif args.num_gpus > 8:
            raise

        if args.debug:
            torch.autograd.detect_anomaly()

        # Prepare experiment directories and save options
        self.project_dir = pathlib.Path(args.project_dir)
        self.experiment_dir = self.project_dir / 'logs' / args.experiment_name
        self.checkpoints_dir = self.experiment_dir / 'checkpoints'
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Redirect stdout
        if args.redirect_print_to_file:
            logs_dir = self.experiment_dir / 'stdout'
            os.makedirs(logs_dir, exist_ok=True)
            sys.stdout = open(os.path.join(logs_dir, f'stdout_{self.rank}.txt'), 'w')
            sys.stderr = open(os.path.join(logs_dir, f'stderr_{self.rank}.txt'), 'w')

        if self.rank == 0:
            print(args)
            with open(self.experiment_dir / 'args.txt', 'wt') as args_file:
                for k, v in sorted(vars(args).items()):
                    args_file.write('%s: %s\n' % (str(k), str(v)))

        # Initialize model
        self.model =  importlib.import_module(f'src.rome_full').TrainableROME(args)

        if args.num_gpus > 0:
            self.model.cuda()

        if self.rank == 0:
            print(self.model)

        # Load pre-trained weights
        if args.model_checkpoint:
            if self.rank == 0:
                print(f'Loading model from {args.model_checkpoint}')
            missing_keys, unexpected_keys = self.model.load_state_dict(
                torch.load(args.model_checkpoint, map_location='cpu'), strict=False)
            print('Missing keys', missing_keys)
            print('Unexpected keys', unexpected_keys)

        # Initialize optimizers and schedulers
        self.opts = self.model.configure_optimizers()

        # Initialize mixed precision
        if args.use_amp:
            self.model, self.opts = amp.initialize(self.model, self.opts, opt_level=args.amp_opt_level,
                                                   num_losses=len(self.opts))

        self.lr_shds, self.lr_shd_max_iters, self.num_iters = self.model.configure_schedulers(self.opts)

        self.use_same_batch_for_all_opts = (
                self.num_iters == [] or
                self.num_iters == [1] * len(self.num_iters)
        )

        if not self.use_same_batch_for_all_opts:
            self.total_num_iters = sum(self.num_iters)

        # Initialize logging
        self.logger = Logger(args, self.experiment_dir, self.rank)

        # Load pre-trained optimizers and schedulers
        if args.trainer_checkpoint:
            if self.rank == 0:
                print(f'Loading trainer from {args.trainer_checkpoint}')
            trainer_checkpoint = torch.load(args.trainer_checkpoint, map_location='cpu')

            for i, opt in enumerate(self.opts):
                opt.load_state_dict(trainer_checkpoint[f'opt_{i}'])

            if len(self.lr_shds):
                for i, shd in enumerate(self.lr_shds):
                    shd.load_state_dict(trainer_checkpoint[f'shd_{i}'])

            if args.use_amp and 'amp' in trainer_checkpoint.keys():
                amp.load_state_dict(trainer_checkpoint['amp'])

            self.logger.load_state_dict(trainer_checkpoint['logger'])

        # Initialize dataloaders
        data_module = importlib.import_module(f'src.dataset.{args.dataset_name}').DataModule(args)

        if args.debug:
            self.train_dataloader, self.train_sampler = data_module.test_dataloader(), None
        else:
            self.train_dataloader, self.train_sampler = data_module.train_dataloader()
        self.test_dataloader = data_module.test_dataloader()

        # Initialize distributed training
        if args.num_gpus > 1:
            self.model = apex.parallel.DistributedDataParallel(self.model)

    @staticmethod
    def get_lr(opt, use_gpu):
        for param_group in opt.param_groups:
            lr = param_group['lr']
            lr = torch.FloatTensor([lr]).mean()
            if use_gpu:
                lr = lr.cuda()

            return lr

    def train(self):
        cur_iter = 0

        for n in range(self.logger.epoch, self.args.max_epochs):
            if self.rank == 0:
                print(f'epoch {n}')
                train_data_iterator = tqdm(self.train_dataloader)
                test_data_iterator = tqdm(self.test_dataloader)
            else:
                train_data_iterator = self.train_dataloader
                test_data_iterator = self.test_dataloader

            # Train
            self.model.train(mode=True)
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(n)

            for data_dict in train_data_iterator:
                losses_dict, histograms_dict, visuals = self.training_step(data_dict, cur_iter)
                cur_iter += 1

                self.logger.log('train', losses_dict, histograms_dict, visuals)

            # Test
            epoch = self.logger.epoch

            if not self.args.skip_test:
                self.model.eval()

                for i, data_dict in enumerate(test_data_iterator):
                    with torch.no_grad():
                        first_batch = i == 0
                        _, losses_dict, histograms_dict, visuals_, _ = self.model(data_dict, visualize=first_batch)
                        if first_batch and epoch % self.args.test_visual_freq == 0:
                            visuals = visuals_  # store visuals from the first batch
                        else:
                            visuals = None
                        self.logger.log('test', losses_dict)

                self.logger.log(
                    'test',
                    histograms_dict=histograms_dict,
                    visuals=visuals,
                    epoch_end=True
                )

            # Save checkpoints
            if self.rank == 0 and (
                    not epoch % self.args.latest_checkpoint_freq or not epoch % self.args.checkpoint_freq):
                # Model
                if self.args.num_gpus > 1:
                    model = self.model.module
                else:
                    model = self.model

                torch.save(model.state_dict(), self.checkpoints_dir / f'{epoch:03d}_model.pth')

                # Trainer
                trainer_checkpoint = {}

                for i, opt in enumerate(self.opts):
                    trainer_checkpoint[f'opt_{i}'] = opt.state_dict()

                if len(self.lr_shds):
                    for i, shd in enumerate(self.lr_shds):
                        trainer_checkpoint[f'shd_{i}'] = shd.state_dict()

                if args.use_amp:
                    trainer_checkpoint['amp'] = amp.state_dict()

                trainer_checkpoint['logger'] = self.logger.state_dict()

                torch.save(trainer_checkpoint, self.checkpoints_dir / f'{epoch:03d}_trainer.pth')

                # Remove previous checkpoint
                prev_epoch = epoch - 1
                if epoch > 1 and prev_epoch % self.args.checkpoint_freq:
                    try:
                        os.remove(self.checkpoints_dir / f'{prev_epoch:03d}_model.pth')
                        os.remove(self.checkpoints_dir / f'{prev_epoch:03d}_trainer.pth')
                    except OSError:
                        pass

    def get_optimizer_idx(self, cur_iter):
        cur_iter_res = cur_iter % self.total_num_iters
        for i in range(len(self.num_iters)):
            cur_iter_res -= self.num_iters[i]
            if cur_iter_res < 0:
                break

        return i

    def forward_backward_step(
            self,
            data_dict,
            losses_dict,
            histograms_dict,
            visuals,
            optimizer_idx,
            output_visuals
    ):
        for i, opt in enumerate(self.opts):
            if i != optimizer_idx:
                # Set requires_grad to False for all other parameters
                for group in opt.param_groups:
                    for p in group['params']:
                        p.requires_grad = False
            else:
                # Set requires_grad to False for all other parameters
                for group in opt.param_groups:
                    for p in group['params']:
                        p.requires_grad = True

        opt = self.opts[optimizer_idx]

        if len(self.lr_shds):
            shd = self.lr_shds[optimizer_idx]
            max_iter = self.lr_shd_max_iters[optimizer_idx]

        opt.zero_grad()

        (
            loss,
            losses_dict_,
            histograms_dict_,
            visuals_,
            data_dict_
        ) = self.model(
            data_dict,
            'train',
            optimizer_idx,
            visualize=output_visuals)

        losses_dict.update(losses_dict_)
        histograms_dict.update(histograms_dict_)
        if visuals_ is not None:
            visuals.data = visuals_.data
        data_dict.update(data_dict_)

        if self.args.use_amp:
            with amp.scale_loss(loss, opt, loss_id=optimizer_idx) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        nan_backward = False
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any().item():
                    nan_backward = True
                    break
        if nan_backward:
            print(f'NaN in Backward, skipping update')
        else:
            opt.step()

        if len(self.lr_shds):
            if shd.last_epoch < max_iter:
                shd.step()

    def training_step(self, data_dict, cur_iter):
        output_visuals = self.logger.output_train_visuals and self.args.output_visuals
        losses_dict = {}
        histograms_dict = {}
        visuals = torch.empty(0)

        if self.use_same_batch_for_all_opts:
            # Use the same batch for all optimizers
            for i in range(len(self.opts)):
                self.forward_backward_step(data_dict, losses_dict, histograms_dict, visuals, i,
                                           output_visuals and i == 0)
        else:
            # Step using a single optimizer
            i = self.get_optimizer_idx(cur_iter)
            self.forward_backward_step(data_dict, losses_dict, histograms_dict, visuals, i, output_visuals)

        if not len(visuals):
            visuals = None

        return losses_dict, histograms_dict, visuals


def main(args):
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--project_dir', default='.', type=str)
    parser.add_argument('--experiment_name', default='', type=str)
    parser.add_argument('--dataset_name', default='', type=str)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--model_checkpoint', default=None, type=str)
    parser.add_argument('--trainer_checkpoint', default=None, type=str)
    parser.add_argument('--torch_home', default='')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--local_rank', type=int)

    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--checkpoint_freq', default=10, type=int)
    parser.add_argument('--latest_checkpoint_freq', default=1, type=int,
                        help='frequency of latest checkpoints creation (in epochs)')
    parser.add_argument('--test_freq', default=1, type=int, help='frequency of testing (in epochs')
    parser.add_argument('--test_visual_freq', default=20, type=int, help='frequency of visuals (in epochs')
    parser.add_argument('--output_visuals', default='True', type=args_utils.str2bool, choices=[True, False])
    parser.add_argument('--logging_freq', default=50, type=int, help='frequency of train logging (in iterations)')
    parser.add_argument('--visuals_freq', default=500, type=int,
                        help='frequency of train visualization (in iterations)')
    parser.add_argument('--redirect_print_to_file', default='False', type=args_utils.str2bool, choices=[True, False])

    parser.add_argument('--use_amp', default='True', type=args_utils.str2bool, choices=[True, False])
    parser.add_argument('--amp_opt_level', default='O0', type=str)

    parser.add_argument('--skip_test', default='False', type=args_utils.str2bool, choices=[True, False])
    parser.add_argument('--debug', action='store_true')

    args, _ = parser.parse_known_args()

    parser = importlib.import_module(f'src.dataset.{args.dataset_name}').DataModule.add_argparse_args(parser)

    parser = importlib.import_module(f'src.rome_full').TrainableROME.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)