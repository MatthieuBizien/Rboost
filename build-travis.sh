#!/usr/bin/env bash

set -ev

cd rboost

# Build and fails on warning
cargo rustc -- -D warnings

# Check the formatting
time cargo fmt -- --check

# Run all the test (in debug mode so it catch more potential errors)
time cargo test --all

# Run all the examples (release mode because it's too slow otherwise)
for f in examples/*; do
    echo "'$f'"
    echo cargo run --example $(basename "$f" | sed 's/.rs//')  --release
    time cargo run --example $(basename "$f" | sed 's/.rs//')  --release
done