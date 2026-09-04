import argparse
import os
import pickle

import torch
from gorilla.config import Config
from tqdm import tqdm

from dataset.ThreeDAPDataset import split_object_data
from utils import *


DEVICE = torch.device('cuda')


def parse_args():
    parser = argparse.ArgumentParser(description="Detect affordance and poses")
    parser.add_argument("--config", help="test config file path")
    parser.add_argument("--checkpoint", help="path to checkpoint model")
    parser.add_argument("--test_data", help="path to test_data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu

    model = build_model(cfg).to(DEVICE)
    if args.checkpoint is None:
        raise ValueError("Must specify a checkpoint path!")

    print("Loading checkpoint....")
    _, exten = os.path.splitext(args.checkpoint)
    if exten == '.t7':
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    elif exten == '.pth':
        check = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(check['model_state_dict'])
    else:
        raise ValueError("Checkpoint must be .t7 or .pth")

    if cfg.get('seed') is not None:
        set_random_seed(cfg.seed)

    with open(args.test_data, 'rb') as f:
        shape_data = pickle.load(f)

    split_seed = cfg.data.get('split_seed', cfg.get('seed', 1))
    shape_data = split_object_data(shape_data, mode='test', split_seed=split_seed)

    # The paper and the published 3DAP evaluation section use 200 poses per
    # affordance-object pair. The stale inherited script used 2000.
    n_sample = cfg.model.get('eval_n_sample', 200)
    guide_w = cfg.model.get('guide_w', 0.2)

    print(f"Detecting with n_sample={n_sample}, guide_w={guide_w}")
    model.eval()
    with torch.no_grad():
        for shape in tqdm(shape_data):
            xyz = torch.from_numpy(shape['full_shape']['coordinate']).unsqueeze(0).float().to(DEVICE)
            shape['result'] = {
                text: [*model.detect_and_sample(xyz, text, n_sample, guide_w=guide_w)]
                for text in shape['affordance']
            }

    with open(f'{cfg.log_dir}/result.pkl', 'wb') as f:
        pickle.dump(shape_data, f)
