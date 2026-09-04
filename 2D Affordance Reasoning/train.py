# -*- coding: utf-8 -*-
"""Standalone training script for the full affordance segmentation model.

All dataset, model, loss, utility, and training definitions are contained in
this file.
"""

import argparse
import json
import os
import random
import time
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPTextModel, CLIPVisionModel


def compute_batch_iou(pred_logits, targets, threshold=0.5):
    preds_bin = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (preds_bin * targets).sum(dim=(2, 3))
    union = preds_bin.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()


def cam_iou_loss(cam_score_map, gt_mask, label, num_classes, eps=1e-6):
    del num_classes
    batch_size, _, height, width = cam_score_map.shape
    gt_mask_down = F.interpolate(gt_mask, size=(height, width), mode="nearest")
    batch_indices = torch.arange(batch_size, device=cam_score_map.device)
    target_scores = cam_score_map[batch_indices, label]
    flat_scores = target_scores.reshape(batch_size, -1)
    min_val = flat_scores.min(dim=1)[0].view(batch_size, 1, 1)
    max_val = flat_scores.max(dim=1)[0].view(batch_size, 1, 1)
    norm_scores = (target_scores - min_val) / (max_val - min_val + eps)
    gt_bin = gt_mask_down.squeeze(1)
    valid = (gt_bin.sum(dim=(1, 2)) > 0).float()
    if valid.sum() == 0:
        return cam_score_map.new_tensor(0.0)
    intersection = (norm_scores * gt_bin).sum(dim=(1, 2))
    union = (norm_scores + gt_bin).sum(dim=(1, 2)) - intersection
    loss = (1 - (intersection + eps) / (union + eps)) * valid
    return loss.sum() / (valid.sum() + eps)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).reshape(-1)
        targets = targets.reshape(-1)
        intersection = (probs * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


def boundary_loss_from_logits(logits, targets):
    probs = torch.sigmoid(logits)
    pred_dx = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
    pred_dy = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
    target_dx = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])
    target_dy = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def cam_seg_consistency_loss(cam_score_map, seg_logits, labels):
    seg_prob = torch.sigmoid(seg_logits)
    cam_prob = F.softmax(cam_score_map, dim=1)
    batch_indices = torch.arange(cam_prob.size(0), device=cam_prob.device)
    selected = cam_prob[batch_indices, labels].unsqueeze(1)
    selected = F.interpolate(
        selected, size=seg_prob.shape[2:], mode="bilinear", align_corners=False
    )
    selected = selected / (selected.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return F.mse_loss(seg_prob, selected)


class HighResEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(1, blocks):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        feat0 = self.conv1(x)
        feat1 = self.maxpool(feat0)
        feat2 = self.layer1(feat1)
        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        return [feat0, feat2, feat3, feat4]


class CrossModalAttention(nn.Module):
    def __init__(self, vis_dim, txt_dim, hidden_dim=256):
        super().__init__()
        self.vis_proj = nn.Linear(vis_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, vis_feat, txt_feat):
        batch_size, channels, _, _ = vis_feat.shape
        vis_flat = vis_feat.flatten(2).transpose(1, 2)
        query = self.txt_proj(txt_feat).unsqueeze(1)
        key = self.vis_proj(vis_flat)
        attention = torch.matmul(query, key.transpose(1, 2)) * self.scale
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, vis_flat)
        output = output.squeeze(1).view(batch_size, channels, 1, 1)
        return vis_feat + output


class TextGuidedDecoder(nn.Module):
    def __init__(self, clip_feat_dim=768, txt_dim=512, out_channels=1):
        super().__init__()
        self.clip_proj = nn.Conv2d(clip_feat_dim, 512, 1)
        self.cross_attn = CrossModalAttention(512, txt_dim)
        self.cam_fuse = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.cam_fuse_28 = nn.Sequential(
            nn.Conv2d(256, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up_convs = nn.ModuleList([
            self._make_up_conv(512, 512),
            self._make_up_conv(256, 256),
            self._make_up_conv(128, 128),
            self._make_up_conv(64, 64),
        ])
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(128, 512, 1),
            nn.Conv2d(64, 256, 1),
            nn.Conv2d(64, 128, 1),
        ])
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(512 + 256, 512, 1),
            nn.Conv2d(512 + 512, 256, 1),
            nn.Conv2d(256 + 256, 128, 1),
            nn.Conv2d(128 + 128, 64, 1),
        ])
        self.ds_convs = nn.ModuleList([
            nn.Conv2d(256, out_channels, 1),
            nn.Conv2d(128, out_channels, 1),
            nn.Conv2d(64, out_channels, 1),
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    @staticmethod
    def _make_up_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, clip_feats, highres_feats, txt_emb):
        clip_deep = self.clip_proj(clip_feats[-1])
        clip_deep = F.interpolate(
            clip_deep, size=(14, 14), mode="bilinear", align_corners=False
        )
        x = self.fusion_convs[0](torch.cat([clip_deep, highres_feats[-1]], dim=1))
        x = self.cross_attn(x, txt_emb)
        cam_feat = self.cam_fuse(x)
        ds_outputs = []

        x = self.up_convs[0](x)
        x = self.fusion_convs[1](torch.cat([x, self.skip_projs[0](highres_feats[-2])], dim=1))
        cam_feat_28 = x
        ds_outputs.append(self.ds_convs[0](x))

        x = self.up_convs[1](x)
        x = self.fusion_convs[2](torch.cat([x, self.skip_projs[1](highres_feats[-3])], dim=1))
        ds_outputs.append(self.ds_convs[1](x))

        x = self.up_convs[2](x)
        x = self.fusion_convs[3](torch.cat([x, self.skip_projs[2](highres_feats[-4])], dim=1))
        ds_outputs.append(self.ds_convs[2](x))

        x = self.up_convs[3](x)
        output = self.final_conv(x)
        cam_feat = cam_feat + F.interpolate(
            self.cam_fuse_28(cam_feat_28),
            size=cam_feat.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return output, ds_outputs, cam_feat


class StrongMultimodalSegModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", num_classes=10,
                 use_cross_attn=True, use_highres=True):
        super().__init__()
        if not use_cross_attn or not use_highres:
            raise ValueError("The standalone full model requires all model components")
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        self.text_projection = nn.Linear(512, 512)
        self.highres_encoder = HighResEncoder()
        self.cam_classifier = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self.seg_decoder = TextGuidedDecoder()
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False
        self._unfreeze_vision_layers(6)

    def _unfreeze_vision_layers(self, num_layers):
        vision_model = self.vision_encoder.vision_model
        for index, layer in enumerate(vision_model.encoder.layers):
            requires_grad = index >= len(vision_model.encoder.layers) - num_layers
            for parameter in layer.parameters():
                parameter.requires_grad = requires_grad
        for parameter in vision_model.embeddings.parameters():
            parameter.requires_grad = True
        for parameter in vision_model.pre_layrnorm.parameters():
            parameter.requires_grad = True

    def forward(self, input_ids, attention_mask, pixel_values):
        text_output = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_embedding = self.text_projection(text_output.pooler_output)
        vision_output = self.vision_encoder(
            pixel_values=pixel_values, output_hidden_states=True, return_dict=True
        )
        clip_feats = []
        for layer_index in [3, 6, 9, 11]:
            hidden_state = vision_output.hidden_states[layer_index][:, 1:, :]
            batch_size, patch_count, channels = hidden_state.shape
            side = int(patch_count ** 0.5)
            clip_feats.append(
                hidden_state.permute(0, 2, 1).reshape(batch_size, channels, side, side)
            )
        highres_feats = self.highres_encoder(pixel_values)
        seg_logits, ds_logits, cam_feat = self.seg_decoder(
            clip_feats, highres_feats, text_embedding
        )
        cam_score_map = self.cam_classifier(cam_feat)
        return cam_score_map.mean(dim=(2, 3)), seg_logits, ds_logits, cam_score_map


class AffordanceSegDataset(Dataset):
    def __init__(self, data_dict, processor, label2id, image_size=224, is_train=False):
        self.data = list(data_dict.values())
        self.processor = processor
        self.label2id = label2id
        self.image_size = image_size
        self.is_train = is_train
        self.mask_resize = T.Resize(
            (image_size, image_size), interpolation=T.InterpolationMode.NEAREST
        )
        self.color_jitter = T.ColorJitter(0.3, 0.3, 0.3, 0.1)
        self.random_grayscale = T.RandomGrayscale(p=0.1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        image = Image.open(item["image_path"]).convert("RGB")
        mask = Image.open(item["mask_path"]).convert("L")
        if self.is_train:
            top, left, height, width = T.RandomResizedCrop.get_params(
                image, scale=(0.7, 1.0), ratio=(0.75, 1.33)
            )
            image = TF.crop(image, top, left, height, width)
            mask = TF.crop(mask, top, left, height, width)
            image = TF.resize(
                image, (self.image_size, self.image_size),
                interpolation=T.InterpolationMode.BILINEAR,
            )
            mask = TF.resize(
                mask, (self.image_size, self.image_size),
                interpolation=T.InterpolationMode.NEAREST,
            )
            if torch.rand(1).item() > 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if torch.rand(1).item() > 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
            angle = torch.randint(-15, 15, (1,)).item()
            if angle:
                image = TF.rotate(
                    image, angle, interpolation=T.InterpolationMode.BILINEAR, fill=0
                )
                mask = TF.rotate(
                    mask, angle, interpolation=T.InterpolationMode.NEAREST, fill=0
                )
            image = self.random_grayscale(self.color_jitter(image))
        else:
            image = TF.resize(
                image, (self.image_size, self.image_size),
                interpolation=T.InterpolationMode.BILINEAR,
            )
            mask = self.mask_resize(mask)
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        )["pixel_values"].squeeze(0)
        mask_tensor = (TF.to_tensor(mask) > 0).float()
        text = self.processor(
            text=item["instruction"], return_tensors="pt", padding="max_length",
            max_length=77, truncation=True,
        )
        return {
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "mask": mask_tensor,
            "label": torch.tensor(self.label2id[item["label"]], dtype=torch.long),
        }


def read_my_data(instruction_id: int, split: str = "train",
                 base_dir: str = "dataset", replicate_single: bool = False) -> Dict:
    dataset_name = "InstructPart"
    split_dir = os.path.join(base_dir, "instructpart_part", dataset_name, split)
    images_dir = os.path.join(split_dir, "images")
    masks_dir = os.path.join(split_dir, "masks")
    json_path = os.path.join(split_dir, f"data_{split}.json")
    data_dict = {}
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    for entry in data:
        base_stem = os.path.splitext(entry["image_path"])[0]
        for item in entry["part_list"]:
            object_name = item["object"].replace(" ", "_")
            part_name = item["part"].replace(" ", "_")
            image_filename = f"{base_stem}-{object_name}-{part_name}.jpg"
            mask_filename = f"{base_stem}-{object_name}-{part_name}.png"
            image_path = os.path.join(images_dir, image_filename)
            mask_path = os.path.join(masks_dir, mask_filename)
            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                continue
            instructions = item.get("instruction", [])
            if isinstance(instructions, str):
                instructions = [instructions]
            if not instructions:
                continue
            if instruction_id is None or instruction_id < 0:
                selected_instructions = instructions
            else:
                selected = instructions[min(instruction_id, len(instructions) - 1)]
                selected_instructions = [selected] * (
                    len(instructions) if replicate_single else 1
                )
            for instruction_index, instruction in enumerate(selected_instructions):
                key = f"{image_filename}__inst{instruction_index}"
                data_dict[key] = {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "instruction": instruction,
                    "label": f"{item['affordance']}_{part_name}",
                }
    return data_dict


def build_label_map(data_dict: Dict) -> Dict[str, int]:
    labels = sorted({item["label"] for item in data_dict.values()})
    return {label: index for index, label in enumerate(labels)}


def save_validation_visualization(
        images, masks, seg_logits, cam_score_maps, save_path, epoch, pred_labels=None):
    os.makedirs(save_path, exist_ok=True)
    if images is None:
        return
    num_samples = min(8, len(images))
    plt.figure(figsize=(16, 4 * num_samples))
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    for index in range(num_samples):
        image = images[index].cpu().permute(1, 2, 0).numpy()
        image = np.clip(image * std + mean, 0, 1)
        mask = masks[index].cpu().squeeze(0).numpy()
        prediction = torch.sigmoid(seg_logits[index]).cpu().squeeze(0).numpy() > 0.5
        cam_map = cam_score_maps[index].max(dim=0)[0].cpu().numpy()
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-6)
        cam_map = np.kron(cam_map, np.ones((16, 16)))[:image.shape[0], :image.shape[1]]
        suffix = (
            f" | Top-1: {pred_labels[index]}"
            if pred_labels is not None and index < len(pred_labels) else ""
        )
        for column, (overlay, title) in enumerate([
            (None, "Input"), (mask, "GT Mask"),
            (prediction, "Pred Mask"), (cam_map, "CAM"),
        ]):
            plt.subplot(num_samples, 4, index * 4 + column + 1)
            plt.imshow(image)
            if overlay is not None:
                plt.imshow(overlay, cmap="jet", alpha=0.5)
            plt.title(f"{title}{suffix}")
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"epoch_{epoch:03d}.png"), dpi=150)
    plt.close()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_full():
    parser = argparse.ArgumentParser(description="Full-model training for affordance segmentation")
    parser.add_argument("--base_dir", type=str, default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    seed_everything(args.seed)

    # Full model uses all instructions and all modules.
    train_data = read_my_data(None, "train", args.base_dir, replicate_single=False)
    val_data = read_my_data(None, "test", args.base_dir)
    label2id = build_label_map(train_data)
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)
    val_data = {k: v for k, v in val_data.items() if v["label"] in label2id}

    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = StrongMultimodalSegModel(
        model_name,
        num_classes,
        use_cross_attn=True,
        use_highres=True,
    )

    train_ds = AffordanceSegDataset(train_data, processor, label2id, is_train=True)
    val_ds = AffordanceSegDataset(val_data, processor, label2id, is_train=False)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    print(
        f"[Info] full model, train samples={len(train_ds)}, val samples={len(val_ds)}, "
        f"num_classes={num_classes}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = [
        {"params": model.cam_classifier.parameters(), "lr": 8e-5},
        {"params": model.seg_decoder.parameters(), "lr": 8e-5},
        {"params": model.highres_encoder.parameters(), "lr": 3e-5},
        {"params": model.vision_encoder.parameters(), "lr": 5e-6},
        {"params": model.text_projection.parameters(), "lr": 5e-6},
    ]
    optimizer = AdamW(params, weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.85,
        patience=5,
        threshold=1e-4,
        min_lr=1e-6,
    )

    criterion_cls = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    scaler = GradScaler(enabled=torch.cuda.is_available())
    max_grad_norm = 0.8
    lambda_ds = 0.15
    lambda_cls = 0.02
    lambda_seg = 1.1
    lambda_cam_base = 0.05
    lambda_boundary = 0.06
    lambda_consistency = 0.03

    patience = 25
    no_improve_epochs = 0
    train_losses, train_accs, train_ious = [], [], []
    val_losses, val_accs, val_ious = [], [], []
    best_val_iou = float("-inf")
    best_result = None

    save_dir = os.path.join("Output", "model-full")
    vis_dir = os.path.join(save_dir, "validation_vis")
    os.makedirs(vis_dir, exist_ok=True)
    total_train_start_time = time.time()

    def is_finite_tensor(x):
        return torch.isfinite(x).all().item()

    def save_best_result(result):
        best_vis_path = os.path.join(vis_dir, f"epoch_{result['best_epoch']:03d}.png")
        lines = [
            f"Variant: {result['variant']}",
            f"Seed: {result['seed']}",
            f"Selection_Metric: Val_IoU",
            f"Best_Epoch: {result['best_epoch']}",
            "",
            f"Train_Loss: {result['train_loss']:.6f}",
            f"Train_Cls_Loss: {result['train_cls_loss']:.6f}",
            f"Train_CAM_Loss: {result['train_cam_loss']:.6f}",
            f"Train_Seg_Loss: {result['train_seg_loss']:.6f}",
            f"Train_Acc: {result['train_acc']:.6f}",
            f"Train_IoU: {result['train_iou']:.6f}",
            "",
            f"Val_Loss: {result['val_loss']:.6f}",
            f"Val_Cls_Loss: {result['val_cls_loss']:.6f}",
            f"Val_CAM_Loss: {result['val_cam_loss']:.6f}",
            f"Val_Seg_Loss: {result['val_seg_loss']:.6f}",
            f"Val_Acc: {result['val_acc']:.6f}",
            f"Val_IoU: {result['val_iou']:.6f}",
            "",
            f"Train_Samples: {len(train_ds)}",
            f"Val_Samples: {len(val_ds)}",
            f"Num_Classes: {num_classes}",
            f"Elapsed_Hours_At_Best: {result['elapsed_hours']:.6f}",
            f"Best_Model: {os.path.join(save_dir, 'best_model.pth')}",
            f"Best_Visualization: {best_vis_path}",
        ]
        with open(os.path.join(save_dir, "best_result.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        t_loss = t_cls_loss = t_cam_loss = t_seg_loss = 0.0
        t_corr = t_total = t_iou = 0
        cam_warmup_epochs = 20
        cam_warmup_ratio = min(1.0, float(epoch + 1) / float(cam_warmup_epochs))
        lambda_cam = lambda_cam_base * cam_warmup_ratio

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} | Train")
        for b in pbar:
            optimizer.zero_grad(set_to_none=True)
            ids = b["input_ids"].to(device, non_blocking=True)
            attn = b["attention_mask"].to(device, non_blocking=True)
            pix = b["pixel_values"].to(device, non_blocking=True)
            msk = b["mask"].to(device, non_blocking=True)
            lab = b["label"].to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                cam_logits, seg_logits, ds_logits, cam_score_map = model(ids, attn, pix)
                loss_cls_raw = criterion_cls(cam_logits, lab)
                loss_cam = cam_iou_loss(cam_score_map, msk, lab, num_classes) if lambda_cam > 0 else seg_logits.new_tensor(0.0)
                loss_seg_main = (
                    criterion_bce(seg_logits, msk)
                    + criterion_dice(seg_logits, msk)
                    + lambda_boundary * boundary_loss_from_logits(seg_logits, msk)
                )
                ds_weights = [0.5, 0.3, 0.2]
                loss_seg_ds = 0.0
                for ds_idx, ds_logit in enumerate(ds_logits):
                    ds_up = F.interpolate(ds_logit, size=msk.shape[2:], mode="bilinear", align_corners=False)
                    ds_weight = ds_weights[min(ds_idx, len(ds_weights) - 1)]
                    ds_loss = (
                        criterion_bce(ds_up, msk)
                        + criterion_dice(ds_up, msk)
                        + lambda_boundary * boundary_loss_from_logits(ds_up, msk)
                    )
                    loss_seg_ds = loss_seg_ds + ds_weight * ds_loss
                loss_seg_raw = loss_seg_main + lambda_ds * loss_seg_ds
                loss_consistency = (
                    cam_seg_consistency_loss(cam_score_map, seg_logits, lab)
                    if lambda_consistency > 0
                    else loss_seg_raw.new_tensor(0.0)
                )
                loss = (
                    lambda_cls * loss_cls_raw
                    + lambda_seg * loss_seg_raw
                    + lambda_cam * loss_cam
                    + lambda_consistency * loss_consistency
                )

            if not (is_finite_tensor(seg_logits) and is_finite_tensor(loss)):
                print(f"[WARN] Non-finite values detected at epoch {epoch}, skipping batch")
                continue

            seg_iou_batch = compute_batch_iou(seg_logits, msk)
            pbar.set_postfix(
                total=f"{loss.item():.3f}",
                cls=f"{loss_cls_raw.item():.3f}",
                cam=f"{loss_cam.item():.3f}",
                seg=f"{loss_seg_raw.item():.3f}",
                iou=f"{seg_iou_batch:.3f}",
            )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            corr = (cam_logits.argmax(1) == lab).sum().item()
            t_corr += corr
            t_total += lab.size(0)
            t_loss += loss.item()
            t_cls_loss += loss_cls_raw.item()
            t_cam_loss += loss_cam.item()
            t_seg_loss += loss_seg_raw.item()
            t_iou += seg_iou_batch * lab.size(0)

        avg_loss = t_loss / max(1, len(train_loader))
        avg_cls_loss = t_cls_loss / max(1, len(train_loader))
        avg_cam_loss = t_cam_loss / max(1, len(train_loader))
        avg_seg_loss = t_seg_loss / max(1, len(train_loader))
        avg_cls_acc = t_corr / max(1, t_total)
        avg_iou = t_iou / max(1, t_total)
        train_losses.append(avg_loss)
        train_accs.append(avg_cls_acc)
        train_ious.append(avg_iou)

        model.eval()
        v_loss = v_cls_loss = v_cam_loss = v_seg_loss = 0.0
        v_corr = v_total = v_iou = 0
        v_img = v_msk_gt = v_seg_logits = v_cam_maps = v_cam_logits = None
        with torch.no_grad():
            for i, b in enumerate(val_loader):
                ids = b["input_ids"].to(device, non_blocking=True)
                attn = b["attention_mask"].to(device, non_blocking=True)
                pix = b["pixel_values"].to(device, non_blocking=True)
                msk = b["mask"].to(device, non_blocking=True)
                lab = b["label"].to(device, non_blocking=True)

                cam_logits, seg_logits, ds_logits, cam_score_map = model(ids, attn, pix)
                loss_cls_raw = criterion_cls(cam_logits, lab)
                loss_cam = cam_iou_loss(cam_score_map, msk, lab, num_classes) if lambda_cam > 0 else seg_logits.new_tensor(0.0)
                loss_seg_main = (
                    criterion_bce(seg_logits, msk)
                    + criterion_dice(seg_logits, msk)
                    + lambda_boundary * boundary_loss_from_logits(seg_logits, msk)
                )
                ds_weights = [0.5, 0.3, 0.2]
                loss_seg_ds = 0.0
                for ds_idx, ds_logit in enumerate(ds_logits):
                    ds_up = F.interpolate(ds_logit, size=msk.shape[2:], mode="bilinear", align_corners=False)
                    ds_weight = ds_weights[min(ds_idx, len(ds_weights) - 1)]
                    ds_loss = (
                        criterion_bce(ds_up, msk)
                        + criterion_dice(ds_up, msk)
                        + lambda_boundary * boundary_loss_from_logits(ds_up, msk)
                    )
                    loss_seg_ds = loss_seg_ds + ds_weight * ds_loss
                loss_seg_raw = loss_seg_main + lambda_ds * loss_seg_ds
                loss_consistency = (
                    cam_seg_consistency_loss(cam_score_map, seg_logits, lab)
                    if lambda_consistency > 0
                    else loss_seg_raw.new_tensor(0.0)
                )
                loss = (
                    lambda_cls * loss_cls_raw
                    + lambda_seg * loss_seg_raw
                    + lambda_cam * loss_cam
                    + lambda_consistency * loss_consistency
                )

                if not (is_finite_tensor(seg_logits) and is_finite_tensor(loss)):
                    print(f"[WARN] Non-finite values detected in validation at epoch {epoch}")
                    continue

                v_loss += loss.item()
                v_cls_loss += loss_cls_raw.item()
                v_cam_loss += loss_cam.item()
                v_seg_loss += loss_seg_raw.item()
                v_corr += (cam_logits.argmax(1) == lab).sum().item()
                v_total += lab.size(0)
                v_iou += compute_batch_iou(seg_logits, msk) * lab.size(0)
                if i == 0:
                    v_img = pix.detach().cpu()
                    v_msk_gt = msk.detach().cpu()
                    v_seg_logits = seg_logits.detach().cpu()
                    v_cam_maps = cam_score_map.detach().cpu()
                    v_cam_logits = cam_logits.detach().cpu()

        avg_vloss = v_loss / max(1, len(val_loader))
        avg_vcls_loss = v_cls_loss / max(1, len(val_loader))
        avg_vcam_loss = v_cam_loss / max(1, len(val_loader))
        avg_vseg_loss = v_seg_loss / max(1, len(val_loader))
        avg_vacc = v_corr / max(1, v_total)
        avg_viou = v_iou / max(1, v_total)
        val_losses.append(avg_vloss)
        val_accs.append(avg_vacc)
        val_ious.append(avg_viou)

        pred_labels = None
        if v_cam_logits is not None:
            pred_ids = v_cam_logits.argmax(dim=1).tolist()
            pred_labels = [id2label.get(i, str(i)) for i in pred_ids]
        save_validation_visualization(v_img, v_msk_gt, v_seg_logits, v_cam_maps, vis_dir, epoch, pred_labels=pred_labels)
        scheduler.step(avg_vloss)

        epoch_time = time.time() - epoch_start_time
        total_train_time = time.time() - total_train_start_time
        print(
            f"[Epoch {epoch:03d}] Train | total={avg_loss:.3f} cls={avg_cls_loss:.3f} cam={avg_cam_loss:.3f} seg={avg_seg_loss:.3f} acc={avg_cls_acc:.3f} iou={avg_iou:.3f} || "
            f"Val | total={avg_vloss:.3f} cls={avg_vcls_loss:.3f} cam={avg_vcam_loss:.3f} seg={avg_vseg_loss:.3f} acc={avg_vacc:.3f} iou={avg_viou:.3f} || "
            f"time={epoch_time:.2f}s total_time={total_train_time / 3600:.2f}h"
        )

        if avg_viou > best_val_iou:
            best_val_iou = avg_viou
            no_improve_epochs = 0
            best_result = {
                "variant": "full",
                "seed": args.seed,
                "best_epoch": epoch,
                "train_loss": avg_loss,
                "train_cls_loss": avg_cls_loss,
                "train_cam_loss": avg_cam_loss,
                "train_seg_loss": avg_seg_loss,
                "train_acc": avg_cls_acc,
                "train_iou": avg_iou,
                "val_loss": avg_vloss,
                "val_cls_loss": avg_vcls_loss,
                "val_cam_loss": avg_vcam_loss,
                "val_seg_loss": avg_vseg_loss,
                "val_acc": avg_vacc,
                "val_iou": avg_viou,
                "elapsed_hours": total_train_time / 3600.0,
            }
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            with open(os.path.join(save_dir, "label2id.json"), "w", encoding="utf-8") as f:
                json.dump(label2id, f, indent=2)
            processor.save_pretrained(save_dir)
            save_best_result(best_result)
            print(f"✅ Best model and result saved (epoch={epoch}, IoU={best_val_iou:.4f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"⏹ Early stopping triggered at epoch {epoch:02d} (no IoU improvement for {patience} epochs)")
                break

    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(
        {
            "Epoch": list(range(len(train_losses))),
            "Train_Loss": train_losses,
            "Train_Acc": train_accs,
            "Train_IoU": train_ious,
            "Val_Loss": val_losses,
            "Val_Acc": val_accs,
            "Val_IoU": val_ious,
        }
    )
    df.to_excel(os.path.join(save_dir, "metrics.xlsx"), index=False)

    total_train_time = time.time() - total_train_start_time
    if best_result is not None:
        best_result["total_train_hours"] = total_train_time / 3600.0
        save_best_result(best_result)
        print(f"训练完成，最佳 epoch: {best_result['best_epoch']}，最佳验证 IoU: {best_val_iou:.4f}")
        print(f"最佳结果已保存至: {os.path.join(save_dir, 'best_result.txt')}")
    else:
        print("训练完成，但没有得到有效的验证结果。")
    print(f"总训练耗时: {total_train_time / 3600:.2f} 小时")


if __name__ == "__main__":
    train_full()
