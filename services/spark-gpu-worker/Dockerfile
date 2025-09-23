FROM bitnami/spark:4.0.0 as sparkdist

FROM nvcr.io/nvidia/jax:25.08-py3
ARG PYTHON_VERSION=3.11.9
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless curl ca-certificates \
    build-essential wget libssl-dev zlib1g-dev libncurses5-dev libffi-dev \
    libsqlite3-dev liblzma-dev tk-dev libbz2-dev libreadline-dev xz-utils && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd / && \
    rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz && \
    apt-get purge -y build-essential wget && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_VERSION=4.0.0
ARG PYSPARK_VERSION=4.0.1
COPY --from=sparkdist /opt/bitnami/spark /opt/spark
ENV SPARK_HOME=/opt/spark
ENV PATH="$SPARK_HOME/bin:$PATH"
RUN /usr/local/bin/python3.11 -m pip install --upgrade pip
ARG JAX_VERSION=0.7.2
ENV PYSPARK_PYTHON=/usr/local/bin/python3.11 PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.11
WORKDIR /
COPY requirements.txt /app/requirements.txt
RUN /usr/local/bin/python3.11 -m pip install --no-cache-dir -r /app/requirements.txt pyspark==${PYSPARK_VERSION} && \
    /usr/local/bin/python3.11 -m pip install --no-cache-dir "jax[cuda12_local]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
COPY getGpusResources.sh /opt/spark/getGpusResources.sh
RUN chmod +x /opt/spark/getGpusResources.sh
COPY --chown=root:root <<'EOF' /entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail
echo "[worker] SPARK_MASTER_URL=${SPARK_MASTER_URL:-}"
if [[ -z "${SPARK_MASTER_URL:-}" ]]; then
    echo "[worker] SPARK_MASTER_URL is not set; defaulting to spark://spark-master:7077"
    SPARK_MASTER_URL="spark://spark-master:7077"
fi
export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=1 -Dspark.worker.resource.gpu.discoveryScript=/opt/spark/getGpusResources.sh ${SPARK_WORKER_OPTS:-}"
echo "[worker] SPARK_WORKER_OPTS=${SPARK_WORKER_OPTS}"
exec "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.worker.Worker \
    "${SPARK_MASTER_URL}"
EOF
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
