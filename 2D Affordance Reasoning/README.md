# 2D Affordance Reasoning

This repository provides a full-model training pipeline for 2D affordance reasoning and part segmentation. The training script combines CLIP-based visual and textual representations, cross-modal attention, high-resolution visual features, CAM supervision, and multi-scale segmentation losses.

The main entry point is [`train.py`](train.py).

## 1. Create the Conda Environment

Create and activate a Python 3.8 environment named `2dar`:

```bash
conda create -n 2dar python=3.8 -y
conda activate 2dar
```

Upgrade `pip` and install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This project requires PyTorch with CUDA support for training. The file defaults to CUDA 12.1. You can modify the torch and torchvision version numbers in requirements.txt according to your machine's actual CUDA version.

## 2. Download and Prepare the Dataset

The training pipeline uses the **InstructPart** dataset from the paper:

> InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning

The complete dataset is available from the following Google Drive folder:

[Download the InstructPart dataset](https://drive.google.com/file/d/14poDpzOlBp8YxVKFMCK-s94NgoDRr7Rm/view?usp=sharing)

Download the complete dataset, extract it, and place it under the project root using the following directory structure:

```text
2D Affordance Reasoning/
├── dataset/
    └── instructpart_part/
        └── InstructPart/
            ├── train/
            │   ├── data_train.json
            │   ├── images/
            │   └── masks/
            └── test/
                ├── data_test.json
                ├── images/
                └── masks/
```

The expected paths are:

```text
dataset/instructpart_part/InstructPart/train/data_train.json
dataset/instructpart_part/InstructPart/train/images/
dataset/instructpart_part/InstructPart/train/masks/
dataset/instructpart_part/InstructPart/test/data_test.json
dataset/instructpart_part/InstructPart/test/images/
dataset/instructpart_part/InstructPart/test/masks/
```

## 3. Download the CLIP Model Weights

The training script uses the Hugging Face model:

`openai/clip-vit-base-patch32`

The model files are available at:

[Download `openai/clip-vit-base-patch32` from Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32/tree/main)

the model directory in the project as follows:

```text
2D Affordance Reasoning/
└── openai/
    └── clip-vit-base-patch32/
        ├── config.json
        ├── preprocessor_config.json
        ├── pytorch_model.bin or model.safetensors
        ├── tokenizer_config.json
        ├── tokenizer.json
        ├── merges.txt
        ├── vocab.json
        └── ...
```

## 4. Run Full-Model Training

From the project root, activate the environment and run:

```bash
python train.py \
    --base_dir dataset \
    --seed 42 \
    --epochs 300
```

## 5. Output Files

Training outputs are written to:

```text
Output/model-full/
```

The main output files and directories are:

```text
Output/model-full/
├── best_model.pth
├── best_result.txt
├── label2id.json
├── metrics.xlsx
├── preprocessor_config.json
├── tokenizer_config.json
├── tokenizer.json
├── merges.txt
├── vocab.json
└── validation_vis/
    ├── epoch_000.png
    ├── epoch_001.png
    └── ...
```

The files contain the following information:

- `best_model.pth`: model parameters from the epoch with the highest validation IoU.
- `best_result.txt`: summary of the best training and validation metrics.
- `label2id.json`: mapping from affordance-part labels to classification IDs.
- `metrics.xlsx`: epoch-level training and validation loss, accuracy, and IoU.
- `validation_vis/`: validation visualizations generated after each epoch.
- Saved processor files: tokenizer and image-preprocessing configuration used by the model.

## 6. Run Full-Model Inference

After training has produced `Output/model-full/best_model.pth`, run the inference script from the project root:

```bash
python infer.py
```

The default command uses the following paths:

| Purpose | Default path |
|---|---|
| Test-set root | `dataset/instructpart_part/InstructPart/test/` |
| Test metadata | `dataset/instructpart_part/InstructPart/test/data_test.json` |
| Test images | `dataset/instructpart_part/InstructPart/test/images/` |
| Test masks | `dataset/instructpart_part/InstructPart/test/masks/` |
| Model checkpoint and metadata | `Output/model-full/` |
| Inference output | `Vis/model-full/` |

The model directory must contain at least:

```text
Output/model-full/
├── best_model.pth
├── label2id.json
├── preprocessor_config.json
├── tokenizer_config.json
├── tokenizer.json
├── merges.txt
└── vocab.json
```

The processor files are normally saved automatically by `train.py`. The inference script loads the checkpoint from `best_model.pth`, loads the label mapping from `label2id.json`, and loads the CLIP processor from `Output/model-full/`.
