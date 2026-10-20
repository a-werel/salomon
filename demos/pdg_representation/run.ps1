$ErrorActionPreference = "Stop"

$examplesPath = Join-Path $PSScriptRoot "examples"
$outputPath = Join-Path $PSScriptRoot "output"

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

docker run --rm -it `
    --mount "type=bind,source=$examplesPath,target=/home/demo/examples,readonly" `
    --mount "type=bind,source=$outputPath,target=/home/demo/output" `
    salomon-pdg

exit $LASTEXITCODE