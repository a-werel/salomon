FROM ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    openjdk-21-jdk-headless \
    python3 \
    sudo \
    gcc \
    g++ \
    graphviz \
    curl \
    unzip \
    ca-certificates \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fL \
    https://github.com/joernio/joern/releases/latest/download/joern-install.sh \
    -o /tmp/joern-install.sh \
    && chmod +x /tmp/joern-install.sh \
    && /tmp/joern-install.sh --install-dir=/opt/joern \
    && rm -f /tmp/joern-install.sh /tmp/joern-cli-*.zip

ENV PATH="/opt/joern/joern-cli:${PATH}"

WORKDIR /home/demo

COPY main.py /home/demo/app/main.py
COPY export_pdg.sc /home/demo/app/export_pdg.sc

RUN python3 -m venv .venv \
    && .venv/bin/python -m pip install --no-cache-dir questionary

ENV PATH="/home/demo/.venv/bin:${PATH}"

ENTRYPOINT ["python3", "/home/demo/app/main.py"] 