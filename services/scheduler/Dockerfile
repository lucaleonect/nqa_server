FROM bitnami/spark:3.5
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY app /app/app
ENV PYTHONUNBUFFERED=1
CMD ["python3", "-m", "app.main"]


