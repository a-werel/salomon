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
# Expand the path with opt/venv/bin and opt/joern/joern-cli.    
ENV PATH="/opt/venv/bin:/opt/joern/joern-cli:${PATH}"
# Set the environment variables for the python.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VULCNN_HOME=/opt/VulCNN

# Create a virtual environment for the container.
RUN python3 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --no-cache-dir pip==24.0 setuptools==69.5.1 wheel==0.43.0

# Install the python dependencies for the container.
COPY docker/vulcnn/requirements.txt /opt/vulcnn-requirements.txt
RUN python -m pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /opt/vulcnn-requirements.txt

# Install the joern dependencies for the container.
COPY third_party/joern/ /opt/joern/joern-cli/
RUN test "$(cat /opt/joern/joern-cli/VERSION)" = "1.1.1125" \
    && chmod +x /opt/joern/joern-cli/joern*

# Build from the local, pinned checkout. Do not clone a moving branch at build time.
COPY third_party/VulCNN/ /opt/VulCNN/
RUN cd /opt/VulCNN/sent2vec \
    && python -m pip install --no-cache-dir --no-build-isolation . \
    && make -j2 \
    && ln -s /opt/VulCNN/sent2vec/fasttext /usr/local/bin/fasttext \
    && python -m pip check \
    && cd /opt/VulCNN \
    && python -c "import sent2vec, normalization, ImageGeneration, model"

# Joern/Ammonite needs a writable Java home when Docker runs with the host UID.
# Set the environment variable for Java.
ENV HOME=/tmp JAVA_TOOL_OPTIONS="-Duser.home=/tmp"
# Copy the cli.py and functions.sc files to the container.
COPY docker/vulcnn/cli.py docker/vulcnn/functions.sc /opt/salomon-vulcnn/
# Create the workspace and models directories.
RUN mkdir -p /workspace /models \
    && chmod 1777 /workspace 
# Set the permissions for the workspace directory.
# Set the working directory for the container.
WORKDIR /workspace
# Set the entrypoint for the container.
# Entrypoint is the command that will be executed when the container is started.
ENTRYPOINT ["python", "/opt/salomon-vulcnn/cli.py"]
# Extra arguments provided to the entrypoint will be passed to the cli.py script and replace the help command.
CMD ["--help"]
