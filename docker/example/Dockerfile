FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install the system dependencies for the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl graphviz openjdk-11-jre-headless \
    python3 python3-dev python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

# Set the environment variables for the container.
ENV VIRTUAL_ENV=/opt/venv
# Set the environment variables for the python.
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EXAMPLE_HOME=/opt/example

# Create a virtual environment for the container.
RUN python3 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --no-cache-dir pip==24.0 setuptools==69.5.1 wheel==0.43.0

# Install the python dependencies for the container.
COPY docker/example/requirements.txt /opt/example-requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/example-requirements.txt

# Build from the local, pinned checkout. Do not clone a moving branch at build time.
COPY third_party/example/ /opt/example/

# Copy the cli.py files to the container.
COPY docker/example/cli.py /opt/salomon-example/
# Create the workspace and models directories.
RUN mkdir -p /workspace /models \
    && chmod 1777 /workspace 
# Set the permissions for the workspace directory.
# Set the working directory for the container.
WORKDIR /workspace
# Set the entrypoint for the container.
# Entrypoint is the command that will be executed when the container is started.
ENTRYPOINT ["python", "/opt/salomon-example/cli.py"]
# Extra arguments provided to the entrypoint will be passed to the cli.py script and replace the help command.
CMD ["--help"]