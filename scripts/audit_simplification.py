"""Verify exported predictions and compute paired success differences in Docker."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from run_simplification import reference, head_inputs
from tsn.common.config import read_json, write_json
from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.splits import episode_catalog
from tsn.evaluation.closed_loop import evaluate_rollouts
from tsn.features.state import policy_state
from tsn.models.consensus_policy import ConsensusPolicy
from tsn.models.simplified_consensus import compact_head, load_compact_policy


def paired(candidate, baseline):
    left = {r['episode_id']: r for r in candidate['results']}
    right = {r['episode_id']: r for r in baseline['results']}
    assert left.keys() == right.keys() and len(left) == 100
    changes = np.array([int(left[k]['success'])-int(right[k]['success']) for k in sorted(left)])
    rng = np.random.default_rng(20261003)
    boot = changes[rng.integers(0, len(changes), (10000, len(changes)))].mean(1)*100
    return dict(success_difference_pp=float(changes.mean()*100),
                paired_bootstrap_95_ci_pp=np.quantile(boot, [.025, .975]).tolist(),
                gained_episodes=int((changes == 1).sum()), lost_episodes=int((changes == -1).sum()),
                episodes=len(changes), bootstrap_resamples=10000)


def benchmark(head, args):
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(10):
            head(*args)
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            start = time.perf_counter()
            head(*args)
            torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
    return dict(median_ms=float(np.median(times)), mean_ms=float(np.mean(times)), repeats=100, batch_size=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline-results', type=Path, default=Path(
        '/run/user/1016/experiments/gtsn_consensus_20261003_fresh_interim_20261003_141728'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = read_json(args.root/'results.json')
    variant = results['deployed_variant']
    reference_model, config, splits = reference(args.root)
    head = compact_head(reference_model, variant)
    head.load_state_dict(torch.load(args.root/'heads'/variant/'best.pt', map_location='cuda', weights_only=True))
    expected_policy = ConsensusPolicy(reference_model.backbone, head, reference_model.kinematics,
                                     servo_radius=.08, execute=15, orientation='baseline').cuda().eval()
    policy, maps, checkpoint = load_compact_policy(args.root/'simplified.pt', 'cuda')
    data_root = Path(config['benchmark']['root'])
    dataset = FrameDataset(data_root, splits['validation'][:1], episode_catalog(data_root), 30, include_rgb=True)
    max_difference = 0.
    # Exercise a real episode's history and a reset on both independent models.
    with torch.inference_mode():
        for frame in (0, 15, 30, 45, 0):
            if frame == 0:
                policy.reset_episode()
                expected_policy.reset_episode()
            item = dataset[min(frame, len(dataset)-1)]
            state = policy_state(item['qpos'][None].cuda(), item['goal_pose'][None].cuda(), maps.settings)
            call = (item['rgb'][None].cuda(), state, item['K'][None].cuda(), item['T_B_C'][None].cuda())
            policy.observe_step(frame)
            expected_policy.observe_step(frame)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                expected = expected_policy(*call)
                actual = policy(*call)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            max_difference = max(max_difference, float((actual-expected).abs().max()))
    dataset.close()
    cache = {key: torch.from_numpy(np.load(args.root/'cache/validation'/f'{key}.npy')[:64].copy()).cuda()
             for key in ('tokens', 'geometry', 'pose', 'ages', 'mask', 'state', 'baseline', 'tcp', 'history')}
    values = head_inputs(cache, torch.tensor([45], device='cuda'))
    original_head = compact_head(reference_model, 'control').cuda().eval()
    latency = {name: benchmark(value, values) for name, value in
               [('original_route', original_head), ('retained_route', policy.head)]}
    full_params = sum(p.numel() for p in reference_model.parameters())
    active_params = sum(p.numel() for p in reference_model.backbone.parameters())+sum(p.numel() for p in original_head.parameters())
    deployed_params = sum(p.numel() for p in policy.parameters())
    comparisons = {}
    if (args.root/'rollouts/test'/variant/'closed_loop.json').exists():
        for partition in ('validation', 'test'):
            candidate = read_json(args.root/'rollouts'/partition/variant/'closed_loop.json')
            original = read_json(args.baseline_results/'rollouts'/partition/'state_memory/closed_loop.json')
            control = read_json(args.root/'rollouts'/partition/'control/closed_loop.json')
            comparisons[partition] = dict(vs_original=paired(candidate, original), vs_matched_control=paired(candidate, control))
    catalog = episode_catalog(data_root)
    smoke_ids = [next(ep for ep in splits['test'] if catalog[ep] == route) for route in ('direct', 'over', 'side')]
    smoke = evaluate_rollouts(smoke_ids, catalog, data_root, args.root/'export_smoke',
                              policy, maps, torch.device('cuda'), config['eval'])
    original_test = {r['episode_id']: r for r in read_json(
        args.root/'rollouts/test'/variant/'closed_loop.json')['results']}
    for actual in smoke['results']:
        expected = original_test[actual['episode_id']]
        for key in ('success', 'termination', 'control_steps', 'replans', 'final_position_error_m'):
            assert actual[key] == expected[key], (actual['episode_id'], key, actual[key], expected[key])
        trajectory = np.load(args.root/'export_smoke/episodes'/actual['episode_id']/'trajectory.npz')
        expected_trajectory = np.load(args.root/'rollouts/test'/variant/'episodes'/actual['episode_id']/'trajectory.npz')
        for key in ('qpos', 'T_B_E', 'predicted_joint_targets'):
            np.testing.assert_array_equal(trajectory[key], expected_trajectory[key])
    write_json(args.root/'export_audit.json', dict(passed=True, variant=variant,
        exact_export_prediction_match=True, maximum_action_difference=max_difference,
        history_and_reset_checked=True, full_training_model_parameters=full_params,
        original_active_policy_parameters=active_params, exported_policy_parameters=deployed_params,
        route_latency=latency, paired_comparisons=comparisons,
        checkpoint_bytes=(args.root/'simplified.pt').stat().st_size,
        exported_closed_loop_episodes=smoke_ids, exported_trajectories_match_evaluated_model_exactly=True))
    print(read_json(args.root/'export_audit.json'))


if __name__ == '__main__':
    main()
