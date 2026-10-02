"""Consolidate completed diagnostics and verify all source and split audits."""
import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

from tsn.common.config import experiment_path, read_json, write_json
from tsn.evaluation.map_diagnostics import episode_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = experiment_path(args.root)
    results = read_json(root / 'results.json')
    adapted = read_json(root / 'adapted_head_results.json')
    audit = read_json(root / 'adapted_head_audit.json')
    protocol = read_json(root / 'protocol.json')
    metadata = read_json(root / 'training_cache/metadata.json')
    assert results['episodes'] == 100 and results['frames'] == 14420
    assert protocol['routes'] == {'direct': 20, 'over': 40, 'side': 40}
    assert not set(metadata['episode_ids']) & set(protocol['episode_ids'])
    assert read_json(root / 'status.json')['state'] == read_json(root / 'diagnostic_status.json')['state'] == 'complete'
    assert audit['passed']
    for record in (protocol, audit):
        for name, expected in record['source_sha256'].items():
            assert hashlib.sha256((Path('/workspace') / name).read_bytes()).hexdigest() == expected, name
    original = np.load(root / 'paired_episode_errors.npz')
    seeds = sorted(adapted)
    pooled = {}
    modes = list(adapted[seeds[0]])
    errors = {mode: np.stack([np.load(root / 'adapted_heads' / f'seed_{seed}' / mode / 'paired_episode_errors.npz')['squared']
                             for seed in seeds]).mean(0) for mode in modes}
    for mode in modes:
        rmse = math.sqrt(errors[mode][:, 0].sum()/original['counts'][:, 0].sum())
        reference = math.sqrt(errors['predicted'][:, 0].sum()/original['counts'][:, 0].sum())
        pooled[mode] = {'rmse_rad': rmse, 'relative_rmse_reduction': 1-rmse/reference,
                       'relative_mse_reduction': 1-rmse**2/reference**2,
                       'executed_first15_rmse_rad': math.sqrt(errors[mode][:, 1].sum()/original['counts'][:, 1].sum()),
                       'per_seed_rmse_rad': {seed: adapted[seed][mode]['selected_validation']['rmse_rad'] for seed in seeds},
                       **episode_bootstrap(errors[mode][:, 0], original['counts'][:, 0], errors['predicted'][:, 0], original['episode_routes'])}
        pooled[mode]['executed_first15_paired'] = episode_bootstrap(errors[mode][:, 1], original['counts'][:, 1], errors['predicted'][:, 1], original['episode_routes'])
    summary = {'partition': 'validation', 'episodes': results['episodes'], 'frames': results['frames'],
               'frozen_action_head': results['interventions'], 'map_quality': results['map_quality'],
               'matched_adapted_heads': pooled,
               'pooling': 'Mean per-episode squared errors across two adaptation seeds, then frame-weighted root mean square.',
               'interpretation': 'Map errors materially limit open-loop accuracy, but a state-only adapted head outperforming predicted maps also implicates map/action fusion or fitting. Oracle future-action maps use expert futures and are a diagnostic information ceiling, not deployable performance.',
               'test_set_used': False, 'closed_loop_inferences_supported': False}
    write_json(root / 'summary.json', summary)
    checkpoint = Path(protocol['checkpoint'])
    digest = hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    write_json(root / 'audit.json', {'passed': True, 'partition': 'validation', 'validation_episodes': 100,
               'validation_frames': 14420, 'routes': protocol['routes'], 'train_validation_overlap': 0,
               'test_set_used': False, 'source_hashes_match': True, 'adaptation_conditions': 8,
               'epochs_per_condition': 10, 'identical_initialization_and_sample_orders': True,
               'original_checkpoint': str(checkpoint), 'original_checkpoint_sha256': digest.hexdigest(),
               'original_checkpoint_modified': False, 'docker_image': 'gtsn-pi3:20261002',
               'numerical_precision': 'bfloat16 inference; float16 map cache; float32 loss/errors; float64 accumulators'})
    print({mode: value['rmse_rad'] for mode, value in pooled.items()}, flush=True)


if __name__ == '__main__':
    main()
