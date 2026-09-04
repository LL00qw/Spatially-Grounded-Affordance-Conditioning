# Spatially Grounded Affordance Conditioning for Instruction-Guided 6-DoF Grasping

**Anonymous code release for ICRA 2027 review**

This repository contains the implementation associated with our work on instruction-guided affordance reasoning and 6-DoF grasp pose estimation. The codebase is organized into two main components:

1. **2D Affordance Reasoning** — 2D affordance reasoning and part segmentation from visual and textual inputs.
2. **AGPENet** — 3D affordance re-grounding and affordance-guided 6-DoF grasp pose generation.

Each component has its own environment, dependencies, dataset preparation instructions, training procedure, and inference/evaluation scripts. Please refer to the component-specific `README.md` files for full details.

---

## Repository Structure

```text
.
├── README.md
│
├── 2D Affordance Reasoning/
│   ├── README.md
│   ├── requirements.txt
│   ├── train.py
│   └── infer.py
│
└── AGPENet/
    ├── README.md
    ├── requirements.txt
    ├── train.py
    ├── detect.py
    ├── visualize.py
    ├── assets/
    ├── config/
    ├── dataset/
    ├── models/
    ├── utils/
    ├── .gitignore
    ├── .gitmodules
    └── LICENSE
```

> **Note:** Datasets, pretrained model files, checkpoints, and generated outputs are not necessarily included in the repository. Please follow the preparation instructions below and in the corresponding subdirectory.

---

## 1. 2D Affordance Reasoning

The **2D Affordance Reasoning** component provides a full-model training and inference pipeline for 2D affordance reasoning and part segmentation.

The training pipeline combines:

- CLIP-based visual and textual representations,
- cross-modal attention,
- high-resolution visual features,
- CAM supervision, and
- multi-scale segmentation losses.

The main scripts are:

```text
2D Affordance Reasoning/
├── train.py
└── infer.py
```

For complete instructions, see:

[`2D Affordance Reasoning/README.md`](./2D%20Affordance%20Reasoning/README.md)

---

## 2. AGPENet

**AGPENet** is the 3D component for affordance re-grounding and affordance-guided 6-DoF grasp pose generation.

It jointly models:

- point-level affordance prediction for semantic re-grounding, and
- diffusion-based 6-DoF grasp pose generation.

For complete instructions, see:

[`AGPENet/README.md`](./AGPENet/README.md)

---
