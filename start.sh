#!/usr/bin/env sh
# Build (if needed) and launch the llmfit TUI. Extra args are passed through,
# e.g. `./start.sh --help` or `./start.sh --json`.
set -eu
cd "$(dirname "$0")"

if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo not found. Install Rust from https://rustup.rs and retry." >&2
    exit 1
fi

# ponytail: cargo is incremental, so a no-op rebuild costs well under a second.
cargo build --release -p llmfit
exec ./target/release/llmfit "$@"
