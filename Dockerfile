FROM nvcr.io/nvidia/jax:25.08-py3
WORKDIR /nqa
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt
COPY  .  /nqa