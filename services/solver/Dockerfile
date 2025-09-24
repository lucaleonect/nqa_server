FROM nvcr.io/nvidia/jax:25.08-py3
WORKDIR /nqa

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY nqa /nqa/nqa
COPY app /nqa/app

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]

