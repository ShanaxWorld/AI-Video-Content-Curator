FROM python:3.12-slim
ENV HOST=0.0.0.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg espeak-ng fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app PATH=/home/app/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    CURATOR_DATA_DIR=/home/app/data HF_HOME=/home/app/.cache/huggingface
WORKDIR /home/app/src

COPY --chown=app requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=app . .
EXPOSE 7860
CMD ["python", "app.py"]
