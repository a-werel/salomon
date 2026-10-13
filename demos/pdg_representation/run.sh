#!/bin/bash

mkdir -p output/

docker run --rm -it \
    -v "$(pwd)/examples:/home/demo/examples:ro" \
    -v "$(pwd)/output:/home/demo/output" \
    salomon-pdg