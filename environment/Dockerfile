FROM --platform=linux/amd64 python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY wheels/ /wheels/
COPY wheels-sha256.txt requirements-lock.txt /tmp/
RUN cd /wheels && sha256sum -c /tmp/wheels-sha256.txt && pip install --no-cache-dir --no-index --find-links /wheels -r /tmp/requirements-lock.txt && rm -rf /wheels
COPY src/ /app/src/
ENV PYTHONPATH=/app/src
