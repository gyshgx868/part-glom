import argparse

from part_seg_reinforce.tools.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        description='Multi-level Part Segmentation Network'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='path to the configuration file'
    )

    # runner
    parser.add_argument(
        '--device',
        type=int,
        default=[0, 1, 2, 3],
        nargs='+',
        help='the indices of GPUs for training or testing'
    )
    parser.add_argument(
        '--phase',
        type=str,
        default='train',
        help='it must be \'train\', or \'test\''
    )
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=32,
        help='training batch size'
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=32,
        help='testing batch size'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=128,
        help='maximum number of training epochs'
    )
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for backbone initialization'
    )
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored during initialization'
    )
    parser.add_argument(
        '--eval-model',
        type=str2bool,
        default=True,
        help='if true, the model will be evaluated during training'
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#epoch)'
    )

    # dataset
    parser.add_argument(
        '--data-path',
        type=str,
        default='./data',
        help='dataset path'
    )
    parser.add_argument(
        '--num-points',
        type=int,
        default=2048,
        help='number of points for each part'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='Chair'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='number of samples'
    )

    # optimizer
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='initial learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGD momentum (default: 0.9)'
    )

    # logging
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results',
        help='path to save results'
    )
    parser.add_argument(
        '--show-details',
        type=str2bool,
        default=True,
        help='whether to show the main classification metrics'
    )
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='whether to print logs or not'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='the interval for printing logs (#iteration)'
    )
    parser.add_argument(
        '--save-model',
        type=str2bool,
        default='true',
        help='if true, the model will be stored'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#epoch)'
    )
    parser.add_argument(
        '--use-tensorboard',
        type=str2bool,
        default='true',
        help='whether to use TensorBoard to visualize results'
    )
    parser.add_argument(
        '--save-cloud',
        type=str2bool,
        default='false',
        help='if true, the point cloud will be stored'
    )

    return parser


def main():
    import json
    p = get_parser()
    js = json.dumps(vars(p), indent=2)
    print(js)


if __name__ == '__main__':
    main()
