FROM ubuntu:24.04

ARG JOERN_VERSION=v4.0.667
ARG TORCH_VERSION=2.15.0
ARG TARGETARCH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       openjdk-21-jdk-headless python3 python3-venv curl unzip ca-certificates gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN case "$TARGETARCH" in \
       amd64) joern_platform=linux-x86_64 ;; \
       arm64) joern_platform=linux-arm64 ;; \
       *) echo "Unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac \
    && curl -fL "https://github.com/joernio/joern/releases/download/${JOERN_VERSION}/joern-cli-${joern_platform}.zip" \
       -o /tmp/joern.zip \
    && unzip -q /tmp/joern.zip -d /opt \
    && rm /tmp/joern.zip

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

RUN /opt/venv/bin/pip install --no-cache-dir numpy==2.5.3

ENV PATH="/opt/venv/bin:/opt/joern-cli:${PATH}" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY src/salomon /app/salomon

# A self-contained image always has a checkpoint, explicitly marked untrained.
RUN python3 -m salomon.adapters.lstm.worker init-demo --output /app/checkpoints/model.pt

ENTRYPOINT ["python3", "-m", "salomon.adapters.lstm.worker"]
