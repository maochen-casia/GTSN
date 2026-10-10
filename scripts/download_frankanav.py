"""Download only FrankaNav metadata and front RGB-D; keep other views untouched."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


def download(root: Path, episodes: list[str], tosutil: str, log_root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        destination = root / episode
        destination.mkdir(exist_ok=True)
        prefix = f'tos://cytoderm-embodied-ai/datasets/frankanav_part1/{episode}'
        for payload in ('data.pkl', 'front', 'front_depth'):
            target = destination / payload
            # A complete local episode (notably 00000) needs no cloud transfer.
            if payload == 'data.pkl' and target.is_file():
                continue
            command = [tosutil, 'cp', f'{prefix}/{payload}', str(target), '-u', '-vchecksum',
                       f'-o={log_root}', f'-cpd={log_root / "checkpoints"}']
            if payload != 'data.pkl':
                # -flat places the contents of this explicit prefix in target.
                command[2] += '/'
                command += ['-r', '-flat', '-j=8']
            log = log_root / f'{episode}_{payload}.log'
            print(f'{episode}: fetching {payload}', flush=True)
            with log.open('w') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            if payload == 'data.pkl' and not target.is_file():
                raise RuntimeError(f'Download produced no metadata: {target}; see {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/datasets_v2/chenmao/frankanav'))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--tosutil', default=shutil.which('tosutil') or '/home/chenmao/tools/tosutil')
    parser.add_argument('--log-root', type=Path, required=True)
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or args.start + args.count > 100:
        parser.error('Select at least one episode within 00000..00099')
    episodes = [f'ep_{index:05d}' for index in range(args.start, args.start + args.count)]
    download(args.root, episodes, args.tosutil, args.log_root.resolve())


if __name__ == '__main__':
    main()
