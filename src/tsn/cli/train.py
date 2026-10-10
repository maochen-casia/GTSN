"""Train the main C1/C2/C3 model without a previous navigation checkpoint."""
import argparse
from pathlib import Path

import torch

from tsn.common.config import PROJECT_ROOT, output_path, read_json
from tsn.training.runner import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/main.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device')
    args = parser.parse_args()
    config = read_json(args.config)
    if args.device:config['train']['device'] = args.device
    options = config['train']
    if any(options[name] <= 0 for name in ('epochs', 'batch_size', 'frame_stride', 'validation_frame_stride', 'draws_per_epoch', 'learning_rate', 'gradient_clip_norm')):
        parser.error('Training sizes, strides, learning rate and gradient limit must be positive')
    if options['num_workers'] < 0 or options['weight_decay'] < 0:
        parser.error('Worker count and weight decay must be nonnegative')
    required = {'route', 'joints', 'maps', 'geometry', 'uncertainty', 'trust'}
    if not required <= options['loss_weights'].keys() or options['loss_weights'].keys()-required-{'memory', 'embodiment'}:
        parser.error('Provide every main-model loss weight')
    if any(value < 0 for value in options['loss_weights'].values()):parser.error('Loss weights must be nonnegative')
    if config['eval']['execute_horizon'] != 15:parser.error('The main policy executes 15 steps')
    torch.set_num_threads(1)
    train(config, output_path(args.output_dir))


if __name__ == '__main__':main()
