FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace
COPY . /workspace

# Core deps for your current scripts
# (vincenty only if you still import baselines/ecr_baseline.py anywhere)
RUN python -m pip install --upgrade pip \
 && python -m pip install \
      scikit-learn \
      matplotlib \
      scipy \
      vincenty \
      mne \
      torcheeg

# Install PyTorch (GPU) on x86_64 DGX (CUDA 12.4 wheels are typically compatible with this container)
# If you only need CPU torch, use: pip install torch torchvision torchaudio
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch torchvision torchaudio

CMD ["bash"]
