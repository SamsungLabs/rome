import torch
import os
import pickle as pkl
import tensorboardX
from torchvision import transforms
import copy
from tqdm import tqdm


class Logger(object):
    def __init__(self, args, experiment_dir, rank):
        super(Logger, self).__init__()
        self.ddp = args.num_gpus > 1
        self.logging_freq = args.logging_freq
        self.visuals_freq = args.visuals_freq
        self.batch_size = args.batch_size
        self.experiment_dir = experiment_dir
        self.checkpoints_dir = self.experiment_dir / 'checkpoints'
        self.rank = rank

        self.train_iter = 0
        self.epoch = 0
        self.output_train_logs = not (self.train_iter + 1) % self.logging_freq
        self.output_train_visuals = self.visuals_freq > 0 and not (self.train_iter + 1) % self.visuals_freq

        self.to_image = transforms.ToPILImage()
        self.losses_buffer = {'train': {}, 'test': {}}

        if self.rank == 0:
            for phase in ['train', 'test']:
                os.makedirs(self.experiment_dir / 'images' / phase, exist_ok=True)

            self.losses = {'train': {}, 'test': {}}
            self.writer = tensorboardX.SummaryWriter(self.experiment_dir)

    def log(self, phase, losses_dict=None, histograms_dict=None, visuals=None, epoch_end=False):
        if losses_dict is not None:
            for name, loss in losses_dict.items():
                if name in self.losses_buffer[phase].keys():
                    self.losses_buffer[phase][name].append(loss)
                else:
                    self.losses_buffer[phase][name] = [loss]

        if phase == 'train':
            self.train_iter += 1

            if self.output_train_logs:
                self.output_logs(phase)
                self.output_histograms(phase, histograms_dict)

            if self.output_train_visuals and visuals is not None:
                self.output_visuals(phase, visuals)

            self.output_train_logs = not (self.train_iter + 1) % self.logging_freq
            self.output_train_visuals = self.visuals_freq > 0 and not (self.train_iter + 1) % self.visuals_freq

        elif phase == 'test' and epoch_end:
            self.epoch += 1
            self.output_logs(phase)

            if visuals is not None:
                self.output_visuals(phase, visuals)

    def output_logs(self, phase):
        # Average the buffers and flush
        names = list(self.losses_buffer[phase].keys())
        losses = []
        for losses_ in self.losses_buffer[phase].values():
            losses.append(torch.stack(losses_).mean())

        if not losses:
            return

        losses = torch.stack(losses)

        self.losses_buffer[phase] = {}

        if self.ddp:
            # Synchronize buffers across GPUs
            losses_ = torch.zeros(size=(torch.distributed.get_world_size(), len(losses)), dtype=losses.dtype,
                                  device=losses.device)
            losses_[self.rank] = losses
            torch.distributed.reduce(losses_, dst=0, op=torch.distributed.ReduceOp.SUM)

            if self.rank == 0:
                losses = losses_.mean(0)

        if self.rank == 0:
            for name, loss in zip(names, losses):
                loss = loss.item()
                if name in self.losses[phase].keys():
                    self.losses[phase][name].append(loss)
                else:
                    self.losses[phase][name] = [loss]

                self.writer.add_scalar(name, loss, self.train_iter)

            tqdm.write(f'Iter {self.train_iter:06d} ' + ', '.join(
                f'{name}: {losses[-1]:.3f}' for name, losses in self.losses[phase].items()))

    def output_histograms(self, phase, histograms):
        if self.rank == 0:
            for key, value in histograms.items():
                value = value.reshape(-1).clone().cpu().data.numpy()
                self.writer.add_histogram(f'{phase}_{key}_hist', value, self.train_iter)

    def output_visuals(self, phase, visuals):
        device = str(visuals.device)

        if self.ddp and device != 'cpu':
            # Synchronize visuals across GPUs
            c, h, w = visuals.shape[1:]
            b = self.batch_size if phase == 'train' else 1
            visuals_ = torch.zeros(size=(torch.distributed.get_world_size(), b, c, h, w), dtype=visuals.dtype,
                                   device=visuals.device)
            visuals_[self.rank, :visuals.shape[0]] = visuals
            torch.distributed.reduce(visuals_, dst=0, op=torch.distributed.ReduceOp.SUM)

            if self.rank == 0:
                visuals = visuals_.view(-1, c, h, w)

        if device != 'cpu':
            # All visuals are reduced, save only one image
            name = f'{self.train_iter:06d}.png'
        else:
            # Save all images
            name = f'{self.train_iter:06d}_{self.rank}.png'

        if self.rank == 0 or device == 'cpu':
            visuals = torch.cat(visuals.split(1, 0), 2)[0]  # cat batch dim in lines w.r.t. height
            visuals = visuals.cpu()

            # Save visuals
            image = self.to_image(visuals)
            image.save(self.experiment_dir / 'images' / phase / name)

            if self.rank == 0:
                self.writer.add_image(f'{phase}_images', visuals, self.train_iter)

    def state_dict(self):
        state_dict = {
            'losses': self.losses,
            'train_iter': self.train_iter,
            'epoch': self.epoch}

        return state_dict

    def load_state_dict(self, state_dict):
        self.losses = state_dict['losses']
        self.train_iter = state_dict['train_iter']
        self.epoch = state_dict['epoch']