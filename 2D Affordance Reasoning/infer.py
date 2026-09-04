# -*- coding: utf-8 -*-
"""Inference and visualization script for the trained full model.

Example:
    python infer.py --base_dir dataset
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPTextModel, CLIPVisionModel


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
            nn.Conv2d(
                in_channels, out_channels, 3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(1, blocks):
            layers.extend([
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False
                ),
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
        x = self.fusion_convs[0](
            torch.cat([clip_deep, highres_feats[-1]], dim=1)
        )
        x = self.cross_attn(x, txt_emb)
        cam_feat = self.cam_fuse(x)
        ds_outputs = []

        x = self.up_convs[0](x)
        x = self.fusion_convs[1](
            torch.cat([x, self.skip_projs[0](highres_feats[-2])], dim=1)
        )
        cam_feat_28 = x
        ds_outputs.append(self.ds_convs[0](x))

        x = self.up_convs[1](x)
        x = self.fusion_convs[2](
            torch.cat([x, self.skip_projs[1](highres_feats[-3])], dim=1)
        )
        ds_outputs.append(self.ds_convs[1](x))

        x = self.up_convs[2](x)
        x = self.fusion_convs[3](
            torch.cat([x, self.skip_projs[2](highres_feats[-4])], dim=1)
        )
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
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        num_classes=10,
        use_cross_attn=True,
        use_highres=True,
    ):
        super().__init__()
        if not use_cross_attn or not use_highres:
            raise ValueError("The full model requires all model components")
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
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        clip_feats = []
        for layer_index in [3, 6, 9, 11]:
            hidden_state = vision_output.hidden_states[layer_index][:, 1:, :]
            batch_size, patch_count, channels = hidden_state.shape
            side = int(patch_count ** 0.5)
            clip_feats.append(
                hidden_state.permute(0, 2, 1).reshape(
                    batch_size, channels, side, side
                )
            )
        highres_feats = self.highres_encoder(pixel_values)
        seg_logits, ds_logits, cam_feat = self.seg_decoder(
            clip_feats, highres_feats, text_embedding
        )
        cam_score_map = self.cam_classifier(cam_feat)
        cam_logits = cam_score_map.mean(dim=(2, 3))
        return cam_logits, seg_logits, ds_logits, cam_score_map


def load_model(model_dir, device):
    label_path = os.path.join(model_dir, "label2id.json")
    checkpoint_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label mapping not found: {label_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    with open(label_path, "r", encoding="utf-8") as file:
        label2id = json.load(file)
    id2label = {int(index): label for label, index in label2id.items()}

    processor = CLIPProcessor.from_pretrained(model_dir)
    model = StrongMultimodalSegModel(
        clip_model_name="openai/clip-vit-base-patch32",
        num_classes=len(label2id),
        use_cross_attn=True,
        use_highres=True,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, processor, id2label


def preprocess(image, instruction, processor, device):
    inputs = processor(
        images=image,
        text=instruction,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True,
    )
    return (
        inputs["pixel_values"].to(device),
        inputs["input_ids"].to(device),
        inputs["attention_mask"].to(device),
    )


def normalized_cam(cam_score_map, class_id, output_size):
    cam = cam_score_map[0, class_id].unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam.cpu().numpy()


def run_inference(model, processor, id2label, image, instruction, device, threshold):
    pixel_values, input_ids, attention_mask = preprocess(image, instruction, processor, device)
    with torch.inference_mode():
        cam_logits, seg_logits, _, cam_score_map = model(input_ids, attention_mask, pixel_values)
        class_probs = F.softmax(cam_logits, dim=1)
        pred_id = int(class_probs.argmax(dim=1).item())
        confidence = float(class_probs[0, pred_id].item())
        seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()

    pred_mask = seg_prob >= threshold
    cam = normalized_cam(cam_score_map, pred_id, seg_prob.shape)
    return {
        "pred_id": pred_id,
        "pred_label": id2label.get(pred_id, str(pred_id)),
        "confidence": confidence,
        "seg_prob": seg_prob,
        "pred_mask": pred_mask,
        "cam": cam,
    }


def resize_array(array, size, nearest=False):
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if array.dtype == np.bool_:
        image = Image.fromarray(array.astype(np.uint8) * 255, mode="L")
        return np.asarray(image.resize(size, resampling)) > 127
    image = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255), mode="L")
    return np.asarray(image.resize(size, resampling), dtype=np.float32) / 255.0


def save_visualization(image, gt_mask, result, instruction, gt_label, save_path):
    image_np = np.asarray(image, dtype=np.uint8)
    width, height = image.size
    gt_np = np.asarray(gt_mask.resize((width, height), Image.Resampling.NEAREST)) > 0
    pred_np = resize_array(result["pred_mask"], (width, height), nearest=True)
    prob_np = resize_array(result["seg_prob"], (width, height))
    cam_np = resize_array(result["cam"], (width, height))

    pred_overlay = image_np.copy()
    pred_overlay[pred_np] = (
        0.5 * pred_overlay[pred_np] + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    figure, axes = plt.subplots(1, 6, figsize=(30, 5))
    panels = [
        (image_np, "Input image", None),
        (gt_np, f"GT mask\n{gt_label}", "gray"),
        (pred_overlay, f"Predicted mask\n{result['pred_label']}", None),
        (prob_np, "Segmentation probability", "viridis"),
        (cam_np, "Predicted-class CAM", "jet"),
        (np.logical_xor(gt_np, pred_np), "Mask error (XOR)", "magma"),
    ]
    for axis, (panel, title, cmap) in zip(axes, panels):
        axis.imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"Instruction: {instruction}\nPrediction confidence: {result['confidence']:.4f}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def safe_name(text):
    return text.replace(" ", "_").replace("/", "_").replace("\\", "_")


def main():
    parser = argparse.ArgumentParser(description="Inference and visualization for the full model")
    parser.add_argument("--base_dir", default="dataset", help="Dataset root directory")
    parser.add_argument(
        "--model_dir",
        default=os.path.join("Output", "model-full"),
        help="Directory containing best_model.pth and label2id.json",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("Vis", "model-full"),
        help="Directory for inference visualizations and result files",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation probability threshold")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of instruction samples to process")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    test_dir = os.path.join(args.base_dir, "instructpart_part", "InstructPart", "test")
    json_path = os.path.join(test_dir, "data_test.json")
    images_dir = os.path.join(test_dir, "images")
    masks_dir = os.path.join(test_dir, "masks")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Test-set JSON not found: {json_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"model_dir={args.model_dir}")
    print(f"test_dir={test_dir}")
    print(f"output_dir={args.output_dir}")
    model, processor, id2label = load_model(args.model_dir, device)

    with open(json_path, "r", encoding="utf-8") as file:
        entries = json.load(file)

    records = []
    processed = 0
    missing = 0
    stop = False
    for entry_index, entry in enumerate(entries):
        base_stem = os.path.splitext(entry["image_path"])[0]
        for part_index, item in enumerate(entry.get("part_list", [])):
            object_name = safe_name(item.get("object", ""))
            part_name = safe_name(item.get("part", ""))
            affordance = item.get("affordance", "")
            image_name = f"{base_stem}-{object_name}-{part_name}.jpg"
            mask_name = f"{base_stem}-{object_name}-{part_name}.png"
            image_path = os.path.join(images_dir, image_name)
            mask_path = os.path.join(masks_dir, mask_name)
            if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
                missing += 1
                print(f"[WARN] Missing image or mask; skipping: {image_name}")
                continue

            instructions = item.get("instruction", [])
            if isinstance(instructions, str):
                instructions = [instructions]
            image = Image.open(image_path).convert("RGB")
            gt_mask = Image.open(mask_path).convert("L")
            sample_dir = os.path.join(
                args.output_dir,
                safe_name(base_stem),
                f"part_{part_index:03d}_{object_name}_{part_name}",
            )
            os.makedirs(sample_dir, exist_ok=True)

            for instruction_index, instruction in enumerate(instructions):
                if args.limit is not None and processed >= args.limit:
                    stop = True
                    break
                result = run_inference(
                    model,
                    processor,
                    id2label,
                    image,
                    instruction,
                    device,
                    args.threshold,
                )
                save_path = os.path.join(sample_dir, f"instruction_{instruction_index:03d}.png")
                gt_label = f"{affordance}_{part_name}"
                save_visualization(image, gt_mask, result, instruction, gt_label, save_path)
                records.append(
                    {
                        "entry_index": entry_index,
                        "part_index": part_index,
                        "instruction_index": instruction_index,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "instruction": instruction,
                        "gt_label": gt_label,
                        "pred_label": result["pred_label"],
                        "pred_confidence": result["confidence"],
                        "visualization": save_path,
                    }
                )
                processed += 1
                print(f"[{processed}] {save_path}")
            if stop:
                break
        if stop:
            break

    results_path = os.path.join(args.output_dir, "inference_results.json")
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("Model: full\n")
        file.write(f"Model_Dir: {args.model_dir}\n")
        file.write(f"Test_JSON: {json_path}\n")
        file.write(f"Threshold: {args.threshold}\n")
        file.write(f"Processed_Instruction_Samples: {processed}\n")
        file.write(f"Missing_Image_Or_Mask_Items: {missing}\n")
    print(f"Inference complete: processed {processed} instruction samples.")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
