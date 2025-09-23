FROM bitnami/spark:3.5
USER root
RUN install_packages python3 python3-pip && \
    pip3 install --no-cache-dir psycopg2-binary SQLAlchemy pyspark
WORKDIR /app
COPY main.py /app/main.py
COPY spark_job.py /app/spark_job.py
COPY spark_job_gpu.py /app/spark_job_gpu.py
ENV PYTHONUNBUFFERED=1
CMD ["python3", "/app/main.py"]