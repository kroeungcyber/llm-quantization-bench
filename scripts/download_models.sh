#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main"

# filename and expected size in bytes (verified via HF API 2026-08-12)
FILES=(
  "Qwen2.5-3B-Instruct-f16.gguf:6178317216"
  "Qwen2.5-3B-Instruct-Q8_0.gguf:3285476512"
  "Qwen2.5-3B-Instruct-Q5_K_M.gguf:2224815264"
  "Qwen2.5-3B-Instruct-Q4_K_M.gguf:1929903264"
)

mkdir -p models

for entry in "${FILES[@]}"; do
  f="${entry%%:*}"
  want="${entry##*:}"
  dest="models/$f"

  if [ -f "$dest" ] && [ "$(stat -f %z "$dest" 2>/dev/null || echo 0)" -eq "$want" ]; then
    echo "SKIP $f (already complete)"
    continue
  fi

  echo "DOWNLOAD $f ($want bytes)"
  curl -L --fail --retry 5 --retry-all-errors -sS -o "$dest" "$BASE_URL/$f"

  got=$(stat -f %z "$dest")
  if [ "$got" -ne "$want" ]; then
    echo "ERROR: $f size mismatch (got $got, want $want)" >&2
    rm -f "$dest"
    exit 1
  fi
  magic=$(head -c 4 "$dest")
  if [ "$magic" != "GGUF" ]; then
    echo "ERROR: $f bad magic ('$magic')" >&2
    rm -f "$dest"
    exit 1
  fi
  echo "OK $f ($got bytes)"
done

echo
ls -lh models/
