# PDG Representation demo

## Overview

Here is a simple overiew of how the PDG representation demo works.

It uses a CLI-tool called joern:
`https://github.com/joernio/joern`

It creates CPG and then translates it to PDG for every internal function in the source code.

## How to run

In order to run the demo, you have do these:
Make sure that you are in the demos/pdg_representation directory. 

```bash
docker build -t salomon-pdg .
```

```bash
./run.sh
```

And that is it! After the execution is finished, in the output folder, there should be `.dot` graphviz files
and `.svg` that you can inspect to see how the PDG will look like.