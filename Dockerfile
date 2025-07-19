FROM pytorch/pytorch

# Install dependencies
RUN pip install --no-cache-dir \
    vllm==0.4.2 \
    pandas==2.2.2

# Copy your script
COPY eval.py /app/math_evaluator.py
COPY utils.py /app/utils.py
COPY data/* /app/data/

WORKDIR /app
ENTRYPOINT ["python", "eval.py"]
