# New image only. The existing demo has CPU-only torch, so use a CUDA runtime.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYOPENGL_PLATFORM=osmesa \
    MPLBACKEND=Agg \
    MS_ASSET_DIR=/opt/maniskill-assets \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONPATH=/workspace/src

# CPU PhysX and software rasterization do not require Vulkan graphics on the host.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 libgl1 libglib2.0-0 libgomp1 libfreetype6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs

RUN python -m pip install --upgrade "setuptools>=68" wheel \
    && python -m pip install . \
    && python -m pip install --no-deps pyrender==0.1.45

# pyrender's stale PyOpenGL==3.1.0 metadata conflicts with modern OSMesa wrappers;
# install it without dependencies after declaring all its actual dependencies above.
# Panda URDF and meshes are shipped in mani-skill; no asset download is required.
CMD ["python", "-m", "tsn.cli.train"]

