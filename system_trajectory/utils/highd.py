import os
import numpy as np
import torch
from torch.utils.data import Dataset

class highdDataset(Dataset):
    def __init__(self, root_path, site_id, split="train", obs_len=20, pred_len=30, kernel_size=3):
        data_dir = os.path.join(root_path, split, str(site_id))
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.kernel_size = kernel_size
        self.data = []

        print(f"[LOADING] highdDataset from: {__file__}")
        print(f"[INIT] Loading data from: {data_dir}")

        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            print(f"[⚠️ WARNING] Dataset folder is missing or empty: {data_dir}")
            return  # Skip dataset creation safely

        for root, _, files in os.walk(data_dir):
            for file in sorted(files):
                if file.endswith('.csv'):
                    path = os.path.join(root, file)
                    raw = np.loadtxt(path, delimiter=',')
                    ids = np.unique(raw[:, 1])
                    for vid in ids:
                        traj = raw[raw[:, 1] == vid]
                        traj = traj[np.argsort(traj[:, 0])]
                        if len(traj) >= obs_len + pred_len:
                            positions = traj[:, 2:4]  # [x, y]
                            lanes = traj[:, 4]
                            for start in range(0, len(traj) - (obs_len + pred_len) + 1):
                                obs = positions[start:start + obs_len]
                                fut = positions[start + obs_len:start + obs_len + pred_len]
                                lane_seq = lanes[start:start + obs_len]
                                self.data.append((obs, fut, lane_seq))

        print(f"[INIT] Total sequences collected: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def is_non_linear(self, traj, threshold=0.002):
        t = np.arange(traj.shape[0])
        fit_x = np.polyval(np.polyfit(t, traj[:, 0], 2), t)
        fit_y = np.polyval(np.polyfit(t, traj[:, 1], 2), t)
        loss = np.mean((traj[:, 0] - fit_x)**2 + (traj[:, 1] - fit_y)**2)
        return float(loss > threshold)

    def __getitem__(self, idx):
        obs_traj, fut_traj, lane_seq = self.data[idx]

        obs_traj = np.asarray(obs_traj)  # [obs_len, 2]
        fut_traj = np.asarray(fut_traj)  # [pred_len, 2]

        obs_traj_rel = obs_traj[1:] - obs_traj[:-1]
        obs_traj_rel = np.vstack([obs_traj_rel[0], obs_traj_rel])  # Keep same length
        fut_traj_rel = fut_traj - obs_traj[-1]

        non_linear_ped = np.array([self.is_non_linear(fut_traj)])
        loss_mask = np.ones((self.pred_len,))

        # Convert to torch tensors, add necessary dims
        obs_traj = torch.tensor(obs_traj, dtype=torch.float32).permute(1, 0).unsqueeze(0)         # [1, 2, 20]
        fut_traj = torch.tensor(fut_traj, dtype=torch.float32).permute(1, 0).unsqueeze(0)         # [1, 2, 30]
        obs_traj_rel = torch.tensor(obs_traj_rel, dtype=torch.float32).permute(1, 0).unsqueeze(0) # [1, 2, 20]
        fut_traj_rel = torch.tensor(fut_traj_rel, dtype=torch.float32).permute(1, 0).unsqueeze(0) # [1, 2, 30]

        V_obs = obs_traj_rel  # Already [1, 2, 20]
        num_nodes = V_obs.shape[1]  # 2 (for 2D trajectory)

        A_obs = torch.eye(num_nodes).unsqueeze(0).repeat(self.kernel_size, 1, 1)  # [K, V, V]
        A_pred = A_obs.clone()

        print(f"[GETITEM] idx: {idx}")
        # print(f"  [0] obs_traj shape       : {obs_traj.shape}")
        # print(f"  [1] fut_traj shape       : {fut_traj.shape}")
        # print(f"  [2] obs_traj_rel shape   : {obs_traj_rel.shape}")
        # print(f"  [3] fut_traj_rel shape   : {fut_traj_rel.shape}")
        # print(f"  [4] non_linear_ped shape : {non_linear_ped.shape}")
        # print(f"  [5] loss_mask shape      : {loss_mask.shape}")
        # print(f"  [6] V_obs shape          : {V_obs.shape}")
        # print(f"  [7] A_obs shape          : {A_obs.shape}")
        # print(f"  [8] V_tr shape (== fut)  : {fut_traj.shape}")
        # print(f"  [9] A_tr shape (== A_obs): {A_pred.shape}")
        print("-" * 40)

        return (
            obs_traj,           # 0
            fut_traj,           # 1
            obs_traj_rel,       # 2
            fut_traj_rel,       # 3
            torch.tensor(non_linear_ped, dtype=torch.float32),  # 4
            torch.tensor(loss_mask, dtype=torch.float32),       # 5
            V_obs,              # 6
            A_obs,              # 7
            fut_traj,           # 8 (V_tr)
            A_pred              # 9 (A_tr)
        )
