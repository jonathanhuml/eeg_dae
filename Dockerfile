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

# CPU-only torch to avoid CUDA/cuDNN library conflicts with TensorFlow container
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio


CMD ["bash"]
