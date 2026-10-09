# Stable foundations first; source changes only rebuild the final stage.
ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE} AS foundation
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

FROM foundation AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PYOPENGL_PLATFORM=osmesa MPLBACKEND=Agg \
    MS_ASSET_DIR=/opt/maniskill-assets \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONPATH=/workspace/src:/workspace/vendor/Pi3 XFORMERS_DISABLED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 libgl1 libglib2.0-0 libgomp1 libfreetype6 libvulkan1 \
    && rm -rf /var/lib/apt/lists/*
# imageio-ffmpeg supplies its encoder; no system ffmpeg or build toolchain.
COPY docker/requirements-runtime.txt /opt/gtsn/requirements-runtime.txt
ARG PACKAGE_INDEX=https://pypi.org/simple
RUN python -m pip install --index-url ${PACKAGE_INDEX} -r /opt/gtsn/requirements-runtime.txt \
    && python -m pip install --index-url ${PACKAGE_INDEX} --no-deps pyrender==0.1.45 \
    && python -c "import torch; assert torch.__version__.split('+')[0] == '2.6.0'; assert torch.version.cuda == '12.4'"
# pyrender declares obsolete PyOpenGL==3.1.0; modern OSMesa needs 3.1.10.
# Its actual dependencies are explicitly installed above.
RUN apt-get update && apt-get install -y --no-install-recommends libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

FROM runtime AS experiment
WORKDIR /workspace
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY vendor/Pi3 ./vendor/Pi3
COPY assets ./assets
COPY scripts ./scripts
COPY tests ./tests
RUN python -m pip install --no-deps --no-build-isolation .
CMD ["python", "-m", "tsn.cli.train", "--help"]
