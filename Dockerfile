FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

# Install Python & dependencies
RUN apt-get update && apt install -y curl

# Install uv and add to PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create and activate venv in single command chain
RUN uv venv --python 3.12 --seed && \
    . .venv/bin/activate
RUN uv pip install vllm --torch-backend=auto
RUN uv pip install pandas

# Copy your script
COPY eval.py /app/eval.py
COPY utils.py /app/utils.py
COPY data/* /app/data/

ENTRYPOINT ["bash"]
