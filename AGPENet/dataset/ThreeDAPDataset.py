import pickle as pkl
import random

from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


def split_object_data(data, mode, split_seed=1):
    """Deterministically apply one shared 70/10/20 *object-level* split.

    The same permutation is recreated for train/val/test so that the three
    subsets are disjoint. Expansion over affordances/poses happens only after
    the object split, matching the paper's protocol.
    """
    if mode not in {"train", "val", "test"}:
        raise ValueError("Mode must be train, val, or test!")

    data = list(data)
    rng = random.Random(split_seed)
    rng.shuffle(data)

    n = len(data)
    n_train = int(0.7 * n)
    n_val = int(0.8 * n)

    if mode == "train":
        return data[:n_train]
    if mode == "val":
        return data[n_train:n_val]
    return data[n_val:]


class ThreeDAPDataset(Dataset):
    """3DAP records expanded after a deterministic object-level split."""

    def __init__(self, data_path, mode, split_seed=1):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.split_seed = split_seed
        self._load_data()

    def _load_data(self):
        self.all_data = []
        with open(self.data_path, "rb") as f:
            data = pkl.load(f)

        data = split_object_data(data, self.mode, self.split_seed)

        # Expansion is deliberately after object splitting.
        for data_point in data:
            for affordance in data_point["affordance"]:
                for pose in data_point["pose"][affordance]:
                    self.all_data.append(
                        {
                            "shape_id": data_point["shape_id"],
                            "semantic class": data_point["semantic class"],
                            "point cloud": data_point["full_shape"]["coordinate"],
                            "affordance": affordance,
                            "affordance label": data_point["full_shape"]["label"][affordance],
                            # scipy uses scalar-last [x,y,z,w]; sign canonicalization
                            # is done immediately before diffusion corruption.
                            "rotation": R.from_matrix(pose[:3, :3]).as_quat(),
                            "translation": pose[:3, 3],
                        }
                    )

    def __getitem__(self, index):
        data_dict = self.all_data[index]
        return (
            data_dict["shape_id"],
            data_dict["semantic class"],
            data_dict["point cloud"],
            data_dict["affordance"],
            data_dict["affordance label"],
            data_dict["rotation"],
            data_dict["translation"],
        )

    def __len__(self):
        return len(self.all_data)


if __name__ == "__main__":
    dataset = ThreeDAPDataset(
        data_path="../full_shape_release.pkl",
        mode="train",
        split_seed=1,
    )
    print(len(dataset))
