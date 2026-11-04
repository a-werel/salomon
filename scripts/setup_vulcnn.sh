#!/usr/bin/env bash
# errors are fatal, unused variables are errors, if command fails in pipe, the whole pipeline fails
set -euo pipefail
# BASH_SOURCE[0] is path to this script
# cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." goes up one level to the repo root
# pwd prints the absolute path of the repo root
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# run git from the repo root, update submodules
git -C "$repo_dir" submodule update --init --recursive
# joern directory
joern_dir="$repo_dir/third_party/joern"
# joern version and sha512 checksum
joern_version=1.1.1125
joern_sha512=4a5150a39d023389883f76caf63d40375fcea5a7d00fbbce78534eeca2950ca6e2fc20d6405c48e9c1a1df563ce485389ada7edb5a287fe84c2023bdb661deba
# if joern is already installed and the version is correct, exit
if [[ -x "$joern_dir/joern-parse" && -f "$joern_dir/VERSION" ]] && [[ "$(cat "$joern_dir/VERSION")" == "$joern_version" ]]; then
    echo "VulCNN initialized; Joern $joern_version is already installed in $joern_dir"
    exit 0
fi
if [[ -e "$joern_dir" ]]; then
    echo "Existing Joern directory is incomplete or has a different version: $joern_dir" >&2
    exit 1
fi
# download directory
download_dir="$(mktemp -d)"
trap 'rm -rf -- "$download_dir"' EXIT
# download joern zip file
curl --fail --location --retry 3 \
    "https://github.com/joernio/joern/releases/download/v$joern_version/joern-cli.zip" \
    --output "$download_dir/joern-cli.zip"
printf '%s  %s\n' "$joern_sha512" "$download_dir/joern-cli.zip" | sha512sum --check -
unzip -q "$download_dir/joern-cli.zip" -d "$download_dir"
printf '%s\n' "$joern_version" > "$download_dir/joern-cli/VERSION"
chmod +x "$download_dir/joern-cli/joern"*
mv -- "$download_dir/joern-cli" "$joern_dir"
echo "Installed Joern $joern_version in $joern_dir"
# copy examples from demo to data raw
demo_dir="$repo_dir/demos"
data_dir="$repo_dir/data"
cp -r "$demo_dir/examples/*" "$data_dir/raw"
echo "Copied examples from demos to data raw"
