# GPU worker. torch + CUDA come from the base image; the host only needs the NVIDIA driver
# and the NVIDIA Container Toolkit.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    PORT=9000

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./

RUN useradd --create-home --uid 10001 appuser && mkdir -p /models && chown -R appuser /srv /models
USER appuser

EXPOSE 9000
# Models take a while to load on first start; /ready flips to 200 when they are in GPU memory.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/ready')" || exit 1

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --no-server-header"]
