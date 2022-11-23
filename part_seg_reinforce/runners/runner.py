import collections
import json
import os
import time
import torch
import yaml

import numpy as np
import torch.nn as nn

from abc import abstractmethod
from torch.utils.data.dataloader import DataLoader


class Runner:
    def __init__(self, args):
        # register variables
        self.model_path = None
        self.tensorboard_path = None
        self.cloud_path = None

        self.dataset = {'train': None, 'test': None}
        self.model1 = None
        self.model2 = None
        self.optimizer1 = None
        self.optimizer2 = None
        self.scheduler1 = None
        self.scheduler2 = None

        self.cur_time = 0
        self.epoch = 0

        # check arguments
        self.args = self.check_args(args)
        args_json = json.dumps(vars(self.args), sort_keys=True, indent=2)
        self.print_log(args_json, print_time=False)

        # devices
        if type(args.device) is list:
            self.output_dev = args.device[0]
            self.print_log(f'{len(args.device)} GPU(s) initialized.')
        else:
            self.output_dev = args.device
            self.print_log(f'1 GPU(s) initialized.')
        # model
        self.load_dataset()
        self.load_model()
        self.load_optimizer()
        self.load_scheduler()
        self.initialize_model()
        # data parallel
        if type(self.args.device) is list and len(self.args.device) > 1:
            self.model1 = nn.DataParallel(
                self.model1, device_ids=args.device,
                output_device=self.output_dev
            )
            self.model2 = nn.DataParallel(
                self.model2, device_ids=args.device,
                output_device=self.output_dev
            )
        if self.dataset['train'] is not None:
            self.dataset['train'] = DataLoader(
                dataset=self.dataset['train'],
                batch_size=self.args.train_batch_size,
                shuffle=True,
                worker_init_fn=np.random.seed(0),
                drop_last=True
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
        args.num_epochs = max(1, args.num_epochs)

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
        self.optimizer1 = torch.optim.SGD(
            self.model1.parameters(),
            lr=self.args.lr * 10,
            weight_decay=1e-5,
            momentum=self.args.momentum
        )
        self.optimizer2 = torch.optim.SGD(
            self.model2.parameters(),
            lr=self.args.lr,
            weight_decay=1e-5,
            momentum=self.args.momentum
        )

    def load_scheduler(self):
        if self.optimizer1 is not None:
            self.scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer1, self.args.num_epochs,
                eta_min=self.args.lr / 100.0
            )
            self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer2, self.args.num_epochs,
                eta_min=self.args.lr / 100.0
            )

    @abstractmethod
    def run(self):
        pass

    def load_model_weights(self, model, weights_file, ignore, key):
        self.print_log(f'Loading model weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        self.epoch = check_points['epoch'] + 1
        # load model weights
        model_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.to(self.output_dev))
            for k, v in check_points[key].items()
        ])
        for w in ignore:
            if model_weights.pop(w, None) is not None:
                self.print_log('Successfully remove weights: {}.'.format(w))
            else:
                self.print_log('Can not remove weights: {}.'.format(w))
        self._try_load_weights(model, model_weights)

    def load_optimizer_weights(self, optimizer, weights_file, key):
        self.print_log(f'Loading optimizer weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        # load optimizer configuration
        optim_weights = check_points[key]
        self._try_load_weights(optimizer, optim_weights)

    def load_scheduler_weights(self, scheduler, weights_file, key):
        self.print_log(f'Loading scheduler weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        # load scheduler configuration
        sched_weights = check_points[key]
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

    def load_weights(self, weights_file, ignore):
        self.load_model_weights(self.model1, weights_file, ignore, key='model1')
        self.load_model_weights(self.model2, weights_file, ignore, key='model2')
        self.load_optimizer_weights(
            self.optimizer1, weights_file, key='optimizer1'
        )
        self.load_optimizer_weights(
            self.optimizer2, weights_file, key='optimizer2'
        )
        self.load_scheduler_weights(
            self.scheduler1, weights_file, key='scheduler1'
        )
        self.load_scheduler_weights(
            self.scheduler2, weights_file, key='scheduler2'
        )

    def save_weights(self, epoch, save_path):
        model_weights1 = collections.OrderedDict([
            (k.split('module.')[-1], v.cpu())
            for k, v in self.model1.state_dict().items()
        ])
        model_weights2 = collections.OrderedDict([
            (k.split('module.')[-1], v.cpu())
            for k, v in self.model2.state_dict().items()
        ])
        save_dict = {
            'epoch': epoch,
            'model1': model_weights1,
            'model2': model_weights2,
            'optimizer1': self.optimizer1.state_dict(),
            'optimizer2': self.optimizer2.state_dict(),
            'scheduler1': self.scheduler1.state_dict(),
            'scheduler2': self.scheduler2.state_dict(),
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
        if print_time:
            localtime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
            msg = "[" + localtime + '] ' + msg
        print(msg)
        if self.args.print_log:
            with open(os.path.join(self.args.save_dir, 'log.txt'), 'a') as f:
                print(msg, file=f)
