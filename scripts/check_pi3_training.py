"""Exercise one real batch and measure Pi3 training speed before the full run."""
import time
from pathlib import Path
import torch

from tsn.common.config import default_config, read_json
from tsn.data.hdf5_dataset import FrameDataset
from tsn.data.splits import make_splits, episode_catalog
from tsn.evaluation.open_loop import device_batch
from tsn.features.state import policy_state
from tsn.models.factory import make_maps, make_policy
from tsn.training.losses import imitation_loss, predicted_map_loss

torch.set_num_threads(1)
benchmark = read_json(default_config('benchmark', 'tsn-1k.json'))
config = read_json(default_config('model', 'pi3_small.json'))
options = read_json(default_config('train', 'pi3_small.json'))
dataset = FrameDataset(Path(benchmark['root']), make_splits(benchmark)['train'][:1],
                       episode_catalog(Path(benchmark['root'])), 30, include_rgb=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=options['batch_size'])
batch = device_batch(next(iter(loader)), torch.device('cuda'))
model = make_policy(config).cuda().train()
training_model = (torch.nn.DataParallel(model, device_ids=options['gpu_ids'])
                  if len(options.get('gpu_ids', [])) > 1 else model)
maps = make_maps(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
state = policy_state(batch['qpos'], batch['goal_pose'], maps.settings)
teacher = maps(batch['depth'], batch['K'], batch['T_B_C'], batch['goal_pose'][:, :3],
               batch['future_ee'], batch['valid_future'])
for iteration in range(4):
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction, predicted_maps = training_model(batch['rgb'], state, batch['K'], batch['T_B_C'], return_maps=True)
        loss = imitation_loss(prediction, batch['target'], batch['valid_future'], .02, 10., .25)
        auxiliary, _ = predicted_map_loss(predicted_maps, teacher)
        loss = loss + .25 * auxiliary
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert torch.isfinite(norm)
    optimizer.step()
    torch.cuda.synchronize()
    print({'iteration': iteration, 'batch_size': len(state), 'seconds': time.monotonic() - started,
           'loss': float(loss.detach()), 'gradient_norm': float(norm),
           'gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9}, flush=True)
print('PI3_REAL_BATCH_PASSED', model.initialization, flush=True)
dataset.close()
