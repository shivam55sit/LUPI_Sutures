import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class PKSutureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: PK_suture_data/
        """
        self.root_dir = root_dir
        # only keep entries that are directories (skip stray files)
        self.patient_ids = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        if len(self.patient_ids) == 0:
            raise ValueError(f"No patient directories found in {root_dir!r}")
        self.transform = transform

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        patient_dir = os.path.join(self.root_dir, pid)

        slit_path = os.path.join(patient_dir, "slitlamp.jpg")
        axial_path = os.path.join(patient_dir, "axial_map.png")

        slit = cv2.imread(slit_path)
        axial = cv2.imread(axial_path)

        # if files are missing on disk, give a clear error before imread failure
        missing_files = []
        if not os.path.exists(slit_path):
            missing_files.append(slit_path)
        if not os.path.exists(axial_path):
            missing_files.append(axial_path)
        if missing_files:
            raise FileNotFoundError(f"Missing files for patient {pid}: {missing_files}")

        if slit is None or axial is None:
            raise RuntimeError(f"Failed to read images for {pid}: {slit_path}, {axial_path}")

        slit = cv2.cvtColor(slit, cv2.COLOR_BGR2RGB)
        axial = cv2.cvtColor(axial, cv2.COLOR_BGR2RGB)

        slit = cv2.resize(slit, (224, 224))
        axial = cv2.resize(axial, (224, 224))

        slit = slit.astype(np.float32) / 255.0
        axial = axial.astype(np.float32) / 255.0

        slit = torch.from_numpy(slit).permute(2, 0, 1)
        axial = torch.from_numpy(axial).permute(2, 0, 1)

        return {
            "slit": slit,
            "axial": axial,
            "patient_id": pid
        }
