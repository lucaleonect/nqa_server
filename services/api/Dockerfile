FROM python:3.13-slim
WORKDIR /app
ARG FASTAPI_VERSION=0.117.1
ARG UVICORN_VERSION=0.37.0
ARG SQLALCHEMY_VERSION=2.0.43
ARG PSYCOPG2_BINARY_VERSION=2.9.10
ARG PYTHON_MULTIPART_VERSION=0.0.20
ARG JINJA2_VERSION=3.1.6
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        fastapi==${FASTAPI_VERSION} \
        "uvicorn[standard]==${UVICORN_VERSION}" \
        SQLAlchemy==${SQLALCHEMY_VERSION} \
        psycopg2-binary==${PSYCOPG2_BINARY_VERSION} \
        python-multipart==${PYTHON_MULTIPART_VERSION} \
        Jinja2==${JINJA2_VERSION}
COPY app /app/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
