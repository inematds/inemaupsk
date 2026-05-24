# Same NGC base as inemaimg — arm64-sbsa, CUDA 13, Blackwell sm_120 kernels.
FROM nvcr.io/nvidia/pytorch:26.03-py3

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY server.py /app/server.py
COPY loaders /app/loaders
COPY web /app/web

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
