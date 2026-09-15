FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements-docker.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements-docker.txt

CMD ["python", "unet.py", "--loss", "jaccard", "--epochs", "1", "--limit", "4", "--image-size", "32", "--batch-size", "1", "--base-filters", "4"]
