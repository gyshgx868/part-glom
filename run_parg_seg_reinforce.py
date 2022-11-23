import random
import torch
import yaml

import numpy as np

from part_seg_reinforce.runners.sample_runner import SampleRunner
from part_seg_reinforce.runners.test_runner import TestRunner
from part_seg_reinforce.runners.train_runner import TrainRunner
from part_seg_reinforce.tools.configuration import get_parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(args).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    args = parser.parse_args()

    if args.phase == 'train':
        runner = TrainRunner(args)
    elif args.phase == 'test':
        runner = TestRunner(args)
    elif args.phase == 'sample':
        runner = SampleRunner(args)
    else:
        raise ValueError('Unknown phase.')

    runner.run()


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
