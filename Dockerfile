# Self-contained compact-model runtime; build under a new project-specific tag.
ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}
ARG INSTALL_RUNTIME=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYOPENGL_PLATFORM=osmesa \
    MPLBACKEND=Agg \
    MS_ASSET_DIR=/opt/maniskill-assets \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONPATH=/workspace/src:/workspace/vendor/Pi3 \
    XFORMERS_DISABLED=1

# CPU PhysX and software rasterization do not require Vulkan graphics on the host.
RUN if [ "$INSTALL_RUNTIME" = 1 ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 libgl1 libglib2.0-0 libgomp1 libfreetype6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*; fi

WORKDIR /workspace
# Clear source baked into an optional project runtime before copying this checkout.
RUN rm -rf /workspace/src /workspace/configs /workspace/scripts /workspace/tests /workspace/vendor
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY vendor/Pi3 ./vendor/Pi3
COPY scripts ./scripts
COPY tests ./tests

RUN if [ "$INSTALL_RUNTIME" = 1 ]; then \
    python -m pip install --upgrade "setuptools>=68" wheel \
    && python -m pip install . \
    && python -m pip install --no-deps pyrender==0.1.45; fi

# pyrender's stale PyOpenGL==3.1.0 metadata conflicts with modern OSMesa wrappers;
# install it without dependencies after declaring all its actual dependencies above.
# Panda URDF and meshes are shipped in mani-skill; no asset download is required.
CMD ["python", "-m", "tsn.cli.train", "--help"]

