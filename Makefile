SHELL := /bin/bash

# Overridable inputs for the inference target
CHECKPOINT ?= models/dynunetclassreg/jan6/checkpoint-58650
OUTPUT_DIR ?= scoring/temp_dynunetclassreg

.PHONY: unetbasic-train dynclassreg-infer package-nnunet-dataset test

# Multi-GPU training with accelerate (DDP)
unetbasic-train:
	set -a && . .env && set +a && \
	PYTHONPATH=$(shell pwd)/src uv run torchrun \
		--nproc_per_node=2 \
		src/approach/unetbasic/train.py \
		--experiment exp_v0

# Single-GPU inference. Paths are relative to the repo root; override on the command line:
#   make dynclassreg-infer CHECKPOINT=models/dynunetclassreg/checkpoint-5000
dynclassreg-infer:
	PYTHONPATH=$(shell pwd)/src uv run python \
		src/approach/unetbasic/inference.py \
		--experiment exp_v0 \
		--checkpoint $(CHECKPOINT) \
		--batch_size 1 \
		--num_workers 4 \
		--output_dir $(OUTPUT_DIR)

package-nnunet-dataset:
	@echo "Copying nnunet_inference files to kaggle_datasets/inference..."
	@cp -v exploration/scoring/nnunet_inference/__init__.py kaggle_datasets/inference/
	@cp -v exploration/scoring/nnunet_inference/inference_nnunet.py kaggle_datasets/inference/
	@cp -v exploration/scoring/post_processing_probs.py kaggle_datasets/inference/
	@cp -v exploration/scoring/postprocess_ribbons.py kaggle_datasets/inference/
	@cp -v exploration/scoring/analyze_prediction_probabilities.py kaggle_datasets/inference/
	@echo "✓ Files copied successfully to kaggle_datasets/inference/"
	@echo "Uploading to Kaggle..."
	cd kaggle_datasets/inference && uv run kaggle datasets version -m 'Update nnunet inference package'
	@echo "✓ Dataset uploaded to Kaggle successfully!"

# Run the test suite
test:
	uv run pytest tests -q
