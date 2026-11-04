# Salomon Docker components

The root `compose.yaml` assembles Salomon using Docker Compose `include`.
It contains only the component list; each component defines its own image,
build, mounts, startup command and health check in its own Compose file.
Docker Compose 2.20.3 or newer is required.

```text
compose.yaml                  # Includes the project's components
docker/
    app/
        compose.yaml          # Salomon app and its model integration settings
        Dockerfile
        Dockerfile.dockerignore
    vulcnn/
        compose.yaml          # VulCNN build and persistent runtime
        Dockerfile
        Dockerfile.dockerignore
        requirements.txt
        cli.py
        functions.sc
third_party/
    VulCNN/                   # Pinned upstream Git submodule
    joern/                    # Downloaded Joern release
```

Salomon-owned Docker files stay outside upstream Git submodules so integrating a
model does not require modifying its upstream checkout. Components can use
different base images, Python versions and dependencies.

## Build and run

Run these commands from the repository root. First prepare the dependencies
required by the currently included model:

```bash
bash scripts/setup_vulcnn.sh
mkdir -p outputs/vulcnn
```

Build all included components, or select one service:

```bash
docker compose build
docker compose build vulcnn
```

Run the Salomon app; Compose starts its declared model dependencies and waits for
their health checks:

```bash
cp demos/pdg_representation/examples/example1.c data/raw/example.c
docker compose run --rm app pdg /data/raw/example.c
docker compose down
```

The app is currently a CLI, so `docker compose up` without an app command displays
its help and exits the app. Model runtimes remain running until stopped.
Classification and trained model requirements are covered in [the VulCNN guide](vulcnn.md).

For component maintenance, VulCNN's Compose file also works on its own:

```bash
docker compose -f docker/vulcnn/compose.yaml build
docker compose -f docker/vulcnn/compose.yaml up -d
docker compose -f docker/vulcnn/compose.yaml down
```

The app's Compose file references model services from the assembled project, so
use the root Compose file to run the app. Avoid running the root and standalone
VulCNN stacks simultaneously: both use the same runtime container name.

## Add another model

1. Add its upstream code under `third_party/` if needed. Put Salomon's container
   definition in `docker/<model>/Dockerfile` and its service definition in
   `docker/<model>/compose.yaml`.
2. Give the service a unique name and image tag. Define its dependencies, mounts,
   startup behavior and health check in that component's Compose file. Keep model
   outputs in `outputs/<model>/` and trained assets in `models/<model>/`.
3. Add `docker/<model>/compose.yaml` to the root `include` list. Compose builds all
   included services with build definitions when you run `docker compose build`.
4. Implement its Salomon adapter and CLI integration. Add any required startup
   dependency, environment variables and shared mounts to
   `docker/app/compose.yaml`. Including a container alone does not add model
   support to the Python app.

For example, after adding a component called `another_model`, the root becomes:

```yaml
include:
  - docker/app/compose.yaml
  - docker/vulcnn/compose.yaml
  - docker/another_model/compose.yaml
```

Paths inside each component's Compose file resolve relative to that file's
directory. For the existing `docker/<component>/` layout, `build.context: ../..`
selects the repository root, and `build.dockerfile` is relative to that build
context. Host mounts such as `../../models` also point back into the repository.
This lets the same component definition work when included from the root or
invoked directly with `-f`.

Each included service must have a unique name. Define each service once in its
own component; `include` does not merge duplicate service definitions.
