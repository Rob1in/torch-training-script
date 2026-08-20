-include .env

VIAM ?= viam
PACKAGE_FILES := model.onnx labels.txt config.yaml pytorch_metrics.json

PACKAGE_DIR = $(RUN_DIR)/onnx_model

.PHONY: upload
upload:
ifndef RUN_DIR
	$(error RUN_DIR is required)
endif
ifndef VERSION
	$(error VERSION is required)
endif
ifndef ORG_ID
	$(error ORG_ID is not set — put it in .env)
endif
ifndef MODEL_NAME
	$(error MODEL_NAME is not set — put it in .env)
endif
	@for f in $(PACKAGE_FILES); do \
		if [ ! -f "$(PACKAGE_DIR)/$$f" ]; then \
			echo "Error: $(PACKAGE_DIR)/$$f is missing — refusing to upload."; \
			echo "Re-run: bash convert_model.sh <run_dir> --dataset-dir <dataset> --pytorch-metrics <metrics.json>"; \
			exit 1; \
		fi; \
	done
	cd "$(PACKAGE_DIR)" && COPYFILE_DISABLE=1 tar -czvf archive.tar.gz $(PACKAGE_FILES)
	$(VIAM) packages upload \
		--org-id=$(ORG_ID) \
		--name=$(MODEL_NAME) \
		--version=$(VERSION) \
		--type=ml_model \
		--model-type=object_detection \
		--path=$(PACKAGE_DIR)/archive.tar.gz \
		--model-framework=onnx
	@echo "Uploaded $(MODEL_NAME):$(VERSION)"