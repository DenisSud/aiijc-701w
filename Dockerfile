FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# -------------------------
# System deps
# -------------------------
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Python deps
# -------------------------
# Core utilities + scientific stack
RUN pip install --no-cache-dir \
    marimo \
    jupyterlab \
    pandas \
    numpy \
    matplotlib \
    scikit-learn \
    tqdm \
    einops \
    sentencepiece \
    datasets \
    transformers

# -------------------------
# Workdir setup
# -------------------------
WORKDIR /workspace

# Default command — open marimo editor
CMD ["marimo", "edit", "main.py"]
