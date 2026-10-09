# Clean experiment runtime on the transferred machine

The image uses PyTorch 2.6.0, CUDA 12.4 and cuDNN 9 runtime libraries. The host
driver supplies GPU access; no CUDA compiler, xformers, desktop renderer, dataset,
weights or experiment outputs are baked into the image. CPU PhysX and OSMesa
provide the current closed-loop environment. imageio-ffmpeg supplies video encoding.
Pi3 uses its PyTorch RoPE fallback; no custom CUDA extension compiler is included.

The Dockerfile has three reusable targets: `foundation` (CUDA/PyTorch), `runtime`
(simulation and Python dependencies), and `experiment` (project source). Runtime
dependencies are installed before source is copied, so code edits reuse those
layers. Keep `docker/requirements-runtime.txt` aligned with `pyproject.toml`.
For future CUDA extension compilation, make a separate development image from a
matching CUDA development base; the current experiment needs only runtime libraries.

Build only under new tags. This host has no working Docker bridge, so builds use
`--network host`; experiment containers remain offline. This machine uses the registry mirror below because
Docker Hub connections time out and the other installed mirror downloads slowly; omit `--build-arg` when Docker Hub is accessible.

```bash
docker build --network host --build-arg BASE_IMAGE=docker.m.daocloud.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  --target foundation -t gtsn-foundation:torch2.6.0-cu124-20261009 .
docker build --network host --build-arg BASE_IMAGE=docker.m.daocloud.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  --build-arg PACKAGE_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  --target runtime -t gtsn-runtime:torch2.6.0-cu124-20261009 .
docker build --network host --build-arg BASE_IMAGE=docker.m.daocloud.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  --build-arg PACKAGE_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t gtsn-experiment:20261009-clean .
```

`pyrender==0.1.45` is deliberately installed without dependency resolution because
its obsolete PyOpenGL pin conflicts with the OSMesa wrappers. Its dependencies
are explicitly installed with PyOpenGL 3.1.10. `pip check` reports that known
metadata mismatch; use rendering smoke checks to validate this combination.

The launcher mounts `/home/datasets_v2/chenmao` read-only at both its real path and
`/run/user/1016`. Existing configurations, recovery provenance and embedded
checkpoint paths therefore continue to work. Override `--storage-root` for another
machine. New writable outputs go under project `runs/`; existing archives stay
read-only. Use a fresh output name for every invocation.

The completed build passed 34 regression tests, a real A800 training forward/backward
and AdamW update, a 15-step closed-loop rollout from the transferred checkpoint,
video encoder execution, and a two-GPU NCCL all-reduce. The rollout is a runtime
smoke check, not a full benchmark evaluation. All 27 pre-existing image records
remain unchanged. Exact image IDs and sizes are in
[`docker/verification.json`](../docker/verification.json); resolved Python versions
are in [`docker/runtime-packages.json`](../docker/runtime-packages.json).

```bash
python3 scripts/docker_run.py --image gtsn-experiment:20261009-clean \
  --gpu 0,1,2,3 --processes 4 train --config configs/cam_var.json \
  --output-dir runs/cam-var-new/training

python3 scripts/docker_run.py --image gtsn-experiment:20261009-clean --gpu 0 evaluate \
  --checkpoint /run/user/1016/experiments/gtsn_cam_var_20261008/full/training/best.pt \
  --partition validation --no-render-videos --output-dir runs/validation-new

python3 scripts/docker_run.py --image gtsn-experiment:20261009-clean --cpu test
```

For replaying an archived implementation, use its frozen source rather than the
current checkout. The current launcher accepts project snapshots; copying the
archive's `source/` into a new directory under project `runs/` allows use of
`--source-snapshot` without editing the archive.
