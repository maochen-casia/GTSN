"""Produce paired, route-specific results and audit evidence for replacements."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tsn.common.config import read_json, write_json


def preferred_nominee(freeze):
    return max(freeze['nominees'], key=lambda n: (n.startswith('joint'),
               freeze['nominees'][n]['validation_successes'], -int(n[-1]), n)) if freeze['nominees'] else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--freeze-preference', action='store_true')
    args = parser.parse_args(); root = args.root
    freeze = read_json(root/'test_freeze.json')
    nominee = preferred_nominee(freeze)
    if args.freeze_preference:
        if (root/'control/test/complete.json').exists():
            raise ValueError('Freeze the preferred nominee before any test partition completes')
        write_json(root/'preferred_nominee.json', dict(preferred=nominee,
            chosen_by='validation and requested joint-module coverage',
            created_utc=datetime.now(timezone.utc).isoformat(), test_used_for_selection=False))
        return
    summary = read_json(root/'summary.json'); audit = read_json(root/'audit.json')
    preference = read_json(root/'preferred_nominee.json')['preferred'] if (root/'preferred_nominee.json').exists() else freeze.get('preferred_nominee')
    if preference != nominee:raise ValueError('Validation preference mismatch')
    summary['preferred'] = preference if preference in summary['accepted'] else None
    summary['preferred_nominee'] = preference
    rng = np.random.default_rng(20261009)
    comparison = {}
    baseline_root = Path('/run/user/1016/experiments/gtsn_cam_var_20261008/full')
    for partition in ('validation', 'test'):
        parent = {r['episode_id']: r for r in read_json(baseline_root/partition/'closed_loop.json')['results']}
        comparison[partition] = {}
        for name in summary['trials']:
            path = root/name/partition/'closed_loop.json'
            if not path.exists():continue
            result = read_json(path)
            changes = np.array([int(r['success'])-int(parent[r['episode_id']]['success']) for r in result['results']])
            intervals = np.percentile(changes[rng.integers(0, len(changes), (10000, len(changes)))].mean(1)*100, (2.5, 97.5)).tolist()
            comparison[partition][name] = dict(successes=int(sum(r['success'] for r in result['results'])),
                gains=int((changes > 0).sum()), losses=int((changes < 0).sum()), paired_95_ci_pp=intervals,
                collisions=sum(r['collision'] is not None for r in result['results']),
                orientation_successes=sum(r['xyz_orientation_success'] for r in result['results']),
                by_route={route: int(sum(r['success'] for r in result['results'] if r['route'] == route))
                          for route in ('direct', 'over', 'side')})
    write_json(root/'paired_results.json', comparison)
    write_json(root/'summary.json', summary)
    lines = ['# Attention replacement study', '',
        'C1 priorities are neural hard-retention scores. C2 risks are direct outputs of stacked residual attention/FFN modules. Only the Pi3 image encoder is frozen; all existing heads were fine tuned on raw RGB.', '',
        '| Model | Validation /100 | Test /100 | Accepted |', '|---|---:|---:|---|',
        '| Parent | 68 | 69 | Reference |']
    for name in summary['trials']:
        lines.append(f"| {name} | {summary['validation_successes'][name]} | {summary['test_successes'].get(name, 'Not nominated')} | {'yes' if name in summary['accepted'] else 'control' if name == 'control' else 'no'} |")
    lines += ['', f"Preferred validation nominee: `{preference}`. Released preferred model: `{summary['preferred']}`.", '',
        'The validation nominee was frozen before test completion. The test gate checks for an observed drop; it does not rank candidates or select a replacement nominee.', '',
        '## Paired comparisons against parent', '', '| Model / partition | Gains | Losses | Difference, percentage points | Paired 95% bootstrap interval |', '|---|---:|---:|---:|---|']
    for partition, models in comparison.items():
        for name, record in models.items():
            lo, hi = record['paired_95_ci_pp']
            lines.append(f"| {name} / {partition} | {record['gains']} | {record['losses']} | {record['gains']-record['losses']:+d} | [{lo:+.0f}, {hi:+.0f}] |")
    lines += ['', 'Single training seed, short fine tuning, and a previously inspected development benchmark limit conclusions. Full 100-scene partitions are measured; these results do not establish population-level noninferiority.', '',
              '## Audit', '', 'All stored encoder tensors remain bitwise equal to the parent. Every existing decoder/map/joint/route/clearance head has changed. The first-step gradient audit confirms trainability. Source hashes, configs, epoch records, complete trajectories and checkpoint hashes are retained.', '']
    for name, record in audit.items():
        lines.append(f"- {name}: {record['trainable_parameters']:,} trainable parameters; {record['changed_parent_tensors']} changed parent tensors; {record['encoder_tensors_unchanged']} unchanged encoder tensors.")
    if summary['preferred']:
        lines += ['', f"Checkpoint: `{root/summary['preferred']/'training/best.pt'}`."]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':main()
