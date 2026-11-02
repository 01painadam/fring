#!/usr/bin/env bash
# Builds the web app into apps/fring-web/dist/.
#
#   1. cargo builds fring-web for wasm32 (profile wasm-release, see the root Cargo.toml).
#   2. wasm-bindgen writes the JavaScript glue and the final .wasm file.
#   3. index.html is copied next to them.
#
# wasm-bindgen-cli must have the same version as the wasm-bindgen crate:
#   cargo install wasm-bindgen-cli --version 0.2.130 --locked
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$(cd "$here/../.." && pwd)"
dist="$here/dist"
want="0.2.130"

if ! command -v wasm-bindgen >/dev/null; then
  echo "wasm-bindgen not found. Run: cargo install wasm-bindgen-cli --version $want --locked" >&2
  exit 1
fi
have="$(wasm-bindgen --version | awk '{print $2}')"
if [ "$have" != "$want" ]; then
  echo "wasm-bindgen $have found, but the crate needs $want. Run: cargo install wasm-bindgen-cli --version $want --locked --force" >&2
  exit 1
fi

cd "$root"
cargo build -p fring-web --lib --target wasm32-unknown-unknown --profile wasm-release

rm -rf "$dist"
mkdir -p "$dist"
wasm-bindgen --target web --no-typescript --out-dir "$dist" \
  "target/wasm32-unknown-unknown/wasm-release/fring_web.wasm"
cp "$here/index.html" "$dist/index.html"

ls -l "$dist"
echo "wasm size: $(wc -c < "$dist/fring_web_bg.wasm") bytes"
