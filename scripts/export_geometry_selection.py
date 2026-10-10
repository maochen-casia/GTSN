"""Export selected or compact adapter weights in a short-lived CPU process."""
import argparse
from pathlib import Path
from tsn.common.checkpoint import load_checkpoint, save_checkpoint
from run_geometry_study import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-successes', type=int)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    if args.output.exists():parser.error('Output checkpoint already exists')
    saved = load_checkpoint(args.input)
    if args.compact:
        if args.parent is None:parser.error('Compact export requires its frozen parent')
        keys = set(load_checkpoint(args.parent)['model'])
        saved = dict(format_version=1, component=saved['component'], parent_checkpoint=str(args.parent),
            parent_sha256=digest(args.parent), config=saved['config'],
            model={key: value for key, value in saved['model'].items() if key not in keys})
    else:
        if args.validation_successes is None:parser.error('Selection requires validation successes')
        saved['validation_successes'] = args.validation_successes
        saved['selected_by'] = 'full validation closed-loop success'
    save_checkpoint(args.output, saved)


if __name__ == '__main__':main()
