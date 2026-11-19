FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_replication.txt /app/
RUN pip install --no-cache-dir -r requirements_replication.txt

COPY run_independent_replication.py /app/
RUN mkdir -p /app/output

ENTRYPOINT ["python", "/app/run_independent_replication.py", "--out-dir", "/app/output"]