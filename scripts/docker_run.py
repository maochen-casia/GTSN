"""Run hand-clearance training, evaluation, reporting, or tests in Docker."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='gtsn-compact:latest')
    parser.add_argument('--gpu', default='all', help='GPU ID, comma-separated IDs, or all')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--print-command', action='store_true')
    parser.add_argument('--source-snapshot', type=Path,
                        help='Run an existing source snapshot within the project or experiment storage')
    parser.add_argument('command', choices=('train', 'evaluate', 'report', 'geometry_figure', 'corrective', 'geometric_energy', 'geometric_energy_report', 'c1_report', 'c1_verify', 'uncertainty', 'c3_report', 'c2_verify', 'c2_report', 'test'))
    args, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    snapshot = args.source_snapshot.resolve() if args.source_snapshot else root
    if args.source_snapshot and (not (snapshot/'src/tsn').is_dir() or
            not any(snapshot.is_relative_to(base) for base in (root, Path('/run/user/1016/experiments')))):
        parser.error('Source snapshot must contain src/tsn within the project or experiment storage')
    command = ['docker', 'run', '--rm', '--init', '--network', 'none', '--read-only',
               '--user', f'{os.getuid()}:{os.getgid()}', '--shm-size', '4g',
               '--tmpfs', '/tmp:rw,exec,size=2g']
    python_path = '/workspace/src:/workspace/vendor/Pi3' if snapshot == root else f'{snapshot}/src:/workspace/vendor/Pi3'
    for setting in ('PYTHONDONTWRITEBYTECODE=1', f'PYTHONPATH={python_path}',
                    'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                    'LP_NUM_THREADS=1', 'XDG_CACHE_HOME=/tmp/cache', 'MPLCONFIGDIR=/tmp/matplotlib'):
        command += ['--env', setting]
    command += ['--mount', f'type=bind,source={root},target=/workspace,readonly']
    if root != Path('/workspace'):
        command += ['--mount', f'type=bind,source={root},target={root},readonly']
    storage = Path('/run/user/1016')
    if storage.is_dir():
        command += ['--mount', f'type=bind,source={storage},target={storage},readonly']
    if args.command != 'test':
        output_parser = argparse.ArgumentParser(add_help=False)
        output_parser.add_argument('--output-dir', type=Path)
        parsed, _ = output_parser.parse_known_args(extra)
        if parsed.output_dir is not None:
            output = parsed.output_dir.resolve()
            allowed = (root / 'runs', storage / 'experiments')
            if not any(output.is_relative_to(base) and output != base for base in allowed):
                parser.error('Output must be a new directory under project runs/ or /run/user/1016/experiments/')
            index = next((i for i, value in enumerate(extra) if value == '--output-dir'), None)
            if index is not None:
                extra[index + 1] = str(output)
            else:
                extra = [f'--output-dir={output}' if value.startswith('--output-dir=') else value for value in extra]
            if output.exists():
                parser.error('Output directory already exists')
            if not args.print_command:
                output.mkdir(parents=True)
            command += ['--mount', f'type=bind,source={output},target={output}']
        if not args.cpu:
            command += ['--gpus', 'all' if args.gpu == 'all' else f'"device={args.gpu}"']
        if args.cpu:
            extra += ['--device', 'cpu']
        invocation = ['python', '-m', f'tsn.cli.{args.command}', *extra]
    else:
        tests = '/workspace/tests' if snapshot == root else str(snapshot/'tests')
        invocation = ['python', '-m', 'unittest', 'discover', '-s', tests, '-v', *extra]
    command += [args.image, *invocation]
    print(shlex.join(command), flush=True)
    if not args.print_command:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
