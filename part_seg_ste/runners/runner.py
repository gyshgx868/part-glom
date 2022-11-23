import collections
import json
import os
import time
import torch
import yaml

import numpy as np
import torch.nn as nn
import torch.distributed as dist

from abc import abstractmethod
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler


class Runner:
    def __init__(self, args):
        # register variables
        self.model_path = None
        self.tensorboard_path = None
        self.cloud_path = None

        self.dataset = {'train': None, 'test': None}
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.cur_time = 0
        self.epoch = 0

        # check arguments
        self.args = self.check_args(args)

        torch.cuda.set_device(args.local_rank)
        dist.init_process_group('nccl')
        args_json = json.dumps(vars(self.args), sort_keys=True, indent=2)
        self.print_log(args_json, print_time=False)
        self.print_log(f'{dist.get_world_size()} GPU(s) initialized.')

        # devices
        self.output_dev = args.local_rank
        if args.train_batch_size % dist.get_world_size() != 0:
            raise ValueError(
                'Please specify the batch size as a multiple of {}'.format(
                    dist.get_world_size()
                )
            )
        # model
        self.load_dataset()
        self.load_model()
        self.load_optimizer()
        self.load_scheduler()
        self.initialize_model()
        # data parallel
        if self.model is not None:
            # convert nn.BatchNorm to nn.SyncBatchNorm
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.output_dev],
                output_device=self.output_dev,
            )
        if self.dataset['train'] is not None:
            self.dataset['train'] = DataLoader(
                dataset=self.dataset['train'],
                batch_size=self.args.train_batch_size // dist.get_world_size(),
                # shuffle=True,
                worker_init_fn=np.random.seed(0),
                drop_last=True,
                sampler=DistributedSampler(self.dataset['train'])
            )
        if self.dataset['test'] is not None:
            self.dataset['test'] = DataLoader(
                dataset=self.dataset['test'],
                batch_size=self.args.test_batch_size,
                shuffle=False,
                worker_init_fn=np.random.seed(0),
                drop_last=False
            )

    def check_args(self, args):
        self.model_path = os.path.join(args.save_dir, 'models')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.tensorboard_path = os.path.join(args.save_dir, 'tensorboard')
        if not os.path.exists(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)
        self.cloud_path = os.path.join(args.save_dir, 'point_clouds')
        if not os.path.exists(self.cloud_path):
            os.makedirs(self.cloud_path)

        args.save_interval = max(1, args.save_interval)
        args.eval_interval = max(1, args.eval_interval)
        args.log_interval = max(1, args.log_interval)
        args.num_cycles = max(1, args.num_cycles)
        args.num_epochs = max(1, args.num_epochs)
        args.sampling = args.sampling.lower()
        if args.sampling not in ['categorical', 'softmax']:
            raise ValueError(
                f'Sampling method {args.sampling} is not implemented.'
            )

        # save configuration file
        config_file = os.path.join(args.save_dir, 'config.yaml')
        args_dict = vars(args)
        with open(config_file, 'w') as f:
            yaml.dump(args_dict, f)
        return args

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def initialize_model(self):
        pass

    def load_optimizer(self):
        mid_parameters, top_parameters = [], []
        for k, v in self.model.named_parameters():
            if k.startswith('top_'):
                top_parameters.append(v)
            else:
                mid_parameters.append(v)
        self.optimizer = torch.optim.SGD(
            [
                {'params': top_parameters},
                {'params': mid_parameters, 'lr': self.args.lr * 10.0}
            ],
            lr=self.args.lr,
            weight_decay=1e-5,
            momentum=self.args.momentum
        )

    def load_scheduler(self):
        if self.optimizer is not None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.args.num_epochs,
                eta_min=self.args.lr / 100.0,
            )

    @abstractmethod
    def run(self):
        pass

    def load_model_weights(self, model, weights_file, ignore):
        self.print_log(f'Loading model weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        self.epoch = check_points['epoch'] + 1
        # load model weights
        model_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.to(self.output_dev))
            for k, v in check_points['model'].items()
        ])
        for w in ignore:
            if model_weights.pop(w, None) is not None:
                self.print_log('Successfully remove weights: {}.'.format(w))
            else:
                self.print_log('Can not remove weights: {}.'.format(w))
        self._try_load_weights(model, model_weights)

    def load_optimizer_weights(self, optimizer, weights_file):
        self.print_log(f'Loading optimizer weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        # load optimizer configuration
        optim_weights = check_points['optimizer']
        self._try_load_weights(optimizer, optim_weights)

    def load_scheduler_weights(self, scheduler, weights_file):
        self.print_log(f'Loading scheduler weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        # load scheduler configuration
        sched_weights = check_points['scheduler']
        self._try_load_weights(scheduler, sched_weights)

    def _try_load_weights(self, model, weights):
        try:
            model.load_state_dict(weights)
        except:
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            self.print_log('Can not find these weights:')
            for d in diff:
                self.print_log(d)
            state.update(weights)
            model.load_state_dict(state)

    def save_weights(self, epoch, model, optimizer, scheduler, save_path):
        if dist.get_rank() != 0:
            return
        model_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.cpu())
            for k, v in model.state_dict().items()
        ])
        optim_weights = optimizer.state_dict()
        sched_weights = scheduler.state_dict()
        save_dict = {
            'epoch': epoch,
            'model': model_weights,
            'optimizer': optim_weights,
            'scheduler': sched_weights
        }
        torch.save(save_dict, save_path)
        self.print_log('Model ' + save_path + ' saved.')

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def tick(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, msg, print_time=True):
        if dist.get_rank() != 0:
            return
        if print_time:
            localtime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
            msg = "[" + localtime + '] ' + msg
        print(msg)
        if self.args.print_log:
            with open(os.path.join(self.args.save_dir, 'log.txt'), 'a') as f:
                print(msg, file=f)

    @staticmethod
    def distributed_reduce_mean(tensor):
        result = tensor.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result / dist.get_world_size()

    @staticmethod
    def distributed_concat(tensors):
        result = [tensors.clone() for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(result, tensors)
        return torch.cat(result, dim=0)
