FROM bitnami/spark:4.0.0
USER root
ARG PYTHON_VERSION=3.11.9
RUN install_packages build-essential wget libssl-dev zlib1g-dev libncurses5-dev libffi-dev \
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
ARG PSYCOPG2_BINARY_VERSION=2.9.10
ARG SQLALCHEMY_VERSION=2.0.43
ARG PYSPARK_VERSION=4.0.1
RUN /usr/local/bin/python3.11 -m pip install --upgrade pip
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.11
RUN /usr/local/bin/python3.11 -m pip install --no-cache-dir \
    psycopg2-binary==${PSYCOPG2_BINARY_VERSION} \
    SQLAlchemy==${SQLALCHEMY_VERSION} \
    pyspark==${PYSPARK_VERSION}
WORKDIR /app
COPY main.py /app/main.py
COPY spark_job.py /app/spark_job.py
COPY spark_job_gpu.py /app/spark_job_gpu.py
ENV PYTHONUNBUFFERED=1
CMD ["/usr/local/bin/python3.11", "/app/main.py"]
