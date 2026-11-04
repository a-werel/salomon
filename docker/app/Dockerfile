FROM docker:27-cli AS docker-cli
FROM python:3.12-slim

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY docker/vulcnn/cli.py /app/docker/vulcnn/cli.py
RUN python -m pip install --no-cache-dir .
ENTRYPOINT ["python", "-m", "salomon.main"]
CMD ["--help"]
