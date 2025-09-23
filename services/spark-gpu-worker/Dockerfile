FROM nvcr.io/nvidia/jax:25.08-py3
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_VERSION=3.5.1 \
    HADOOP_VERSION=3
RUN curl -fsSL https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    | tar -xz -C /opt && ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark
ENV SPARK_HOME=/opt/spark
ENV PATH="$SPARK_HOME/bin:$PATH"
WORKDIR /
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt pyspark
COPY getGpusResources.sh /opt/spark/getGpusResources.sh
RUN chmod +x /opt/spark/getGpusResources.sh
CMD bash -lc "exec $SPARK_HOME/bin/spark-class org.apache.spark.deploy.worker.Worker \
    -Dspark.worker.resource.gpu.amount=1 \
    -Dspark.worker.resource.gpu.discoveryScript=/opt/spark/getGpusResources.sh \
    ${SPARK_MASTER_URL}"