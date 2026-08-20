#!/usr/bin/env bash
# Pipeline test for the Makefile `upload` target. Stubs the viam CLI via the
# VIAM variable so nothing is actually uploaded to the registry.
# Usage: bash tests/test_upload.sh   (from anywhere; cd's to the repo root)
set -u

cd "$(dirname "$0")/.."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fails=0

fail() { echo "FAIL: $1"; fails=$((fails + 1)); }
pass() { echo "ok: $1"; }

mkdir -p "$tmp/run/onnx_model"
touch "$tmp/run/onnx_model"/{model.onnx,labels.txt,config.yaml,pytorch_metrics.json}

cat > "$tmp/fake_viam" <<EOF
#!/bin/sh
echo "\$@" > "$tmp/args"
EOF
chmod +x "$tmp/fake_viam"

# --- happy path: all files present, stub CLI receives the right flags ------
if make upload RUN_DIR="$tmp/run" VERSION=0.0-test ORG_ID=fake-org MODEL_NAME=test-model VIAM="$tmp/fake_viam" > "$tmp/out" 2>&1; then
    pass "upload target exits 0"
else
    fail "upload target errored: $(cat "$tmp/out")"
fi

for flag in "--org-id=fake-org" "--name=test-model" "--version=0.0-test" \
            "--type=ml_model" "--model-type=object_detection" "--model-framework=onnx"; do
    grep -q -- "$flag" "$tmp/args" 2>/dev/null && pass "passes $flag" || fail "missing $flag in viam args"
done

test -f "$tmp/run/onnx_model/archive.tar.gz" && pass "archive created" || fail "archive.tar.gz not created"
tar -tzf "$tmp/run/onnx_model/archive.tar.gz" | grep -q model.onnx && pass "archive contains model.onnx" \
    || fail "model.onnx not in archive"
grep -q "Uploaded test-model:0.0-test" "$tmp/out" && pass "prints confirmation" || fail "no confirmation line"

# --- missing package file: must refuse before invoking the CLI -------------
rm "$tmp/run/onnx_model/labels.txt" "$tmp/args"
if make upload RUN_DIR="$tmp/run" VERSION=0.0-test ORG_ID=fake-org MODEL_NAME=test-model VIAM="$tmp/fake_viam" > "$tmp/out" 2>&1; then
    fail "upload succeeded despite missing labels.txt"
else
    pass "refuses upload when labels.txt is missing"
fi
test -f "$tmp/args" && fail "viam CLI was invoked despite missing file" || pass "viam CLI not invoked"

# --- everything present except pytorch_metrics.json ------------------------
touch "$tmp/run/onnx_model/labels.txt"
rm "$tmp/run/onnx_model/pytorch_metrics.json"
if make upload RUN_DIR="$tmp/run" VERSION=0.0-test ORG_ID=fake-org MODEL_NAME=test-model VIAM="$tmp/fake_viam" > "$tmp/out" 2>&1; then
    fail "upload succeeded despite missing pytorch_metrics.json"
else
    pass "refuses upload when pytorch_metrics.json is missing"
fi
test -f "$tmp/args" && fail "viam CLI was invoked despite missing pytorch_metrics.json" || pass "viam CLI not invoked"
grep -q "pytorch-metrics" "$tmp/out" && pass "error points at --pytorch-metrics flag" \
    || fail "error does not mention the --pytorch-metrics remediation"
touch "$tmp/run/onnx_model/pytorch_metrics.json"

# --- missing required variables --------------------------------------------
make upload VERSION=x ORG_ID=fake-org VIAM="$tmp/fake_viam" > /dev/null 2>&1 \
    && fail "succeeded without RUN_DIR" || pass "requires RUN_DIR"
make upload RUN_DIR="$tmp/run" ORG_ID=fake-org VIAM="$tmp/fake_viam" > /dev/null 2>&1 \
    && fail "succeeded without VERSION" || pass "requires VERSION"
make upload RUN_DIR="$tmp/run" VERSION=x ORG_ID=fake-org MODEL_NAME= VIAM="$tmp/fake_viam" > /dev/null 2>&1 \
    && fail "succeeded without MODEL_NAME" || pass "requires MODEL_NAME"

echo
if [ "$fails" -eq 0 ]; then echo "all tests passed"; else echo "$fails test(s) failed"; exit 1; fi
