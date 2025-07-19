FROM vllm/vllm-openai:v0.9.2

# Install dependencies
RUN pip install --no-cache-dir -q pandas

# Copy your script
COPY eval.py /app/eval.py
COPY utils.py /app/utils.py
COPY data/* /app/data/

WORKDIR /app
ENTRYPOINT ["python", "eval.py"]
