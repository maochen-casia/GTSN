"""Check the retained checkpoint against its immutable experiment source.

Run inside the project Docker image. This checks existing weights; it does not
train or select a new model.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

from tsn.common.checkpoint import load_checkpoint
from tsn.models.current_view_policy import CurrentViewHead
from tsn.training.corrective import SequenceCache


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('archived_head', args.source/'src/tsn/models/persistent_policy.py')
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    checkpoint = load_checkpoint(args.checkpoint)
    report = {}
    for label, kind, mixed in (('cpu', 'cpu', False), ('cuda_float32', 'cuda', False),
                                ('cuda_bfloat16', 'cuda', True)):
        device = torch.device(kind)
        legacy = archived.PersistentSceneHead(**checkpoint['memory_options']).to(device).eval()
        legacy.load_state_dict(checkpoint['head'])
        current = CurrentViewHead.from_checkpoint(checkpoint).to(device).eval()
        data = SequenceCache(args.cache/'validation', device)
        worst, count, stream_worst, repeat_worst = 0., 0, 0., 0.
        # CUDA scatter reductions are nondeterministic. BF16 can magnify tiny
        # summation differences to one output quantization step. Check FP32
        # separately and measure the unchanged reference's own repeat error.
        tolerance = 1e-3 if mixed else 2e-6
        for start in range(0, len(data.sequences), 16):
            batch = data.batch(list(range(start, min(start+16, len(data.sequences)))))
            inputs = [batch[k] for k in ('tokens', 'geometry', 'pose', 'frame', 'valid', 'state', 'tcp')]
            with torch.autocast(kind, dtype=torch.bfloat16, enabled=mixed):
                expected = legacy.sequence(*inputs, points=batch['points'])[0]
                repeated = legacy.sequence(*inputs, points=batch['points'])[0]
                actual = current.sequence(*inputs, points=batch['points'])[0]
            mask = batch['valid']
            torch.testing.assert_close(actual[mask], expected[mask], rtol=0, atol=tolerance)
            worst = max(worst, float((actual-expected)[mask].abs().max()))
            repeat_worst = max(repeat_worst, float((repeated-expected)[mask].abs().max()))
            count += int(mask.sum())
        # Exercise deployment cadence and the reduced carry across every frame
        # of representative short and long validation histories.
        selected = sorted(range(len(data.sequences)), key=lambda i: len(data.sequences[i]))[::10]
        for index in selected:
            batch = data.batch([index])
            old_memory, new_memory = None, None
            for t in range(batch['valid'].shape[1]):
                elapsed = batch['frame'][:, t]-(batch['frame'][:, t-1] if t else 0)
                inputs = [batch[k][:, t] for k in ('tokens', 'geometry', 'pose')]
                with torch.autocast(kind, dtype=torch.bfloat16, enabled=mixed):
                    expected, _, old_memory = legacy.stream(*inputs, elapsed, batch['state'][:, t],
                        batch['tcp'][:, t], batch['points'][:, t], old_memory)
                    actual, _, new_memory = current.stream(*inputs, elapsed, batch['state'][:, t],
                        batch['tcp'][:, t], batch['points'][:, t], new_memory)
                torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)
                stream_worst = max(stream_worst, float((actual-expected).abs().max()))
            assert len(new_memory) == 1 and new_memory[0].shape == (1, 4, 258)
        report[label] = dict(observations=count, max_error_m=worst, stream_max_error_m=stream_worst,
                            reference_repeat_max_error_m=repeat_worst,
                            stream_episodes=len(selected), tolerance_m=tolerance)
        print(label, report[label], flush=True)
        del legacy, current, data
    args.output.write_text(json.dumps(dict(passed=True, **report), indent=2)+'\n')


if __name__ == '__main__':
    main()
