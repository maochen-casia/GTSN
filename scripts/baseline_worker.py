"""Train one baseline and freeze its checkpoint before isolated evaluation."""
import argparse
from pathlib import Path
import torch
from tsn.baselines.runner import train
from tsn.common.config import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root, method = args.root, args.method
    directory = root/method
    directory.mkdir()
    config = read_json(root/'configs'/f'{method}.json')
    device = torch.device('cuda')
    torch.cuda.set_device(0)
    # Keep each compact worker's allocator below 8 GiB on the shared GPU.
    torch.cuda.set_per_process_memory_fraction(.1)
    train(config, root/'cache', directory/'training', device)
    write_json(directory/'status.json', {'phase': 'trained'})


if __name__ == '__main__':
    main()
