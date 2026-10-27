SHELL := /bin/bash
.PHONY: setup clean

SCRIPT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO_ROOT  := $(abspath $(SCRIPT_DIR)/../..)
VENV_DIR   := $(REPO_ROOT)/.venv-vision-service
DIST_DIR   := $(REPO_ROOT)/dist

ifeq ($(OS),Windows_NT)
    ACTIVATE := $(VENV_DIR)/Scripts/activate
    BINARY   := $(DIST_DIR)/onnx-vision-service.exe
else
    ACTIVATE := $(VENV_DIR)/bin/activate
    BINARY   := $(DIST_DIR)/onnx-vision-service
endif

# --- Setup ---

.setup: setup.sh
	bash $(SCRIPT_DIR)/setup.sh

setup: .setup

# --- Build ---

# onnxruntime-openvino and openvino ship native libraries that PyInstaller's
# import analysis does not see; --collect-all bundles them. Only packages
# that are actually installed are collected (openvino is Windows-only here).
PYI_COLLECT_CMD := python3 -c "import importlib.util as u; print(' '.join('--collect-all ' + p for p in ('onnxruntime', 'openvino') if u.find_spec(p)))"

$(BINARY): .setup $(SCRIPT_DIR)/main.py $(SCRIPT_DIR)/onnx_vision_service.py $(SCRIPT_DIR)/utils.py $(REPO_ROOT)/omni_inference/core.py
	cd $(REPO_ROOT) && source $(ACTIVATE) && python3 -m PyInstaller \
		--onefile \
		--hidden-import="googleapiclient" \
		$$($(PYI_COLLECT_CMD)) \
		--name onnx-vision-service \
		src/onnx_vision_service/main.py
	rm -rf $(REPO_ROOT)/build $(REPO_ROOT)/onnx-vision-service.spec

dist/archive.tar.gz: $(BINARY)
ifeq ($(OS),Windows_NT)
	cp $(REPO_ROOT)/meta.json $(DIST_DIR)/meta.json
	cd $(DIST_DIR) && tar -czvf archive.tar.gz onnx-vision-service.exe meta.json
else
	cp $(REPO_ROOT)/meta.json $(DIST_DIR)/meta.json
	cd $(DIST_DIR) && tar -czvf archive.tar.gz onnx-vision-service meta.json
endif

# --- Clean ---

clean:
	rm -rf $(DIST_DIR)
	rm -rf $(REPO_ROOT)/build
	rm -f $(REPO_ROOT)/onnx-vision-service.spec
	rm -f $(SCRIPT_DIR)/.setup
