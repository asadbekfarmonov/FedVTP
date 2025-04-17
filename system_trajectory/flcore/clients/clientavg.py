import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
# from utils.privacy import *
from utils.trajectory_utils import *
import torch.nn.functional as F

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                         weight_decay=0.0001)  # todo 换优化器
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        # print("used steplr 0.9")

        # differential privacy
        # self.bad_batch_lst = []
        if self.privacy:
            check_dp(self.model)
            initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

    def train(self, epoch=0):
        if self.graph:
            trainloader = self.load_train_data()
        else:
            trainloader = self.train_samples

        start_time = time.time()
        self.model.train()
        loss_batch = 0
        batch_count = 0
        is_fst_loss = True
        loader_len = len(trainloader)
        turn_point = int(loader_len / self.batch_size) * self.batch_size + loader_len % self.batch_size - 1

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            batch_count = 0
            for cnt, batch in enumerate(trainloader):
                batch_count += 1
                if torch.cuda.is_available():
                    batch = [tensor.cuda() if isinstance(tensor, torch.Tensor) else tensor for tensor in batch]
                else:
                    batch = [tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor for tensor in batch]
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
                loss_mask, V_obs, A_obs, V_tr, A_tr = batch

                self.optimizer.zero_grad()

                print(f"[LOG] V_obs: {V_obs.shape} | {V_obs.dtype}")
                V_obs_tmp = Client.format_V_obs(V_obs)  # CORRECT
                print(f"[LOG] V_obs_tmp (permute): {V_obs_tmp.shape} | {V_obs_tmp.dtype}")
                print(f"[LOG] A_obs (permute): {A_obs.shape} | {A_obs.dtype}")
                A_fixed = A_obs.squeeze()
                V_pred, _ = self.model(V_obs_tmp, A_fixed)
                print(f"[LOG] V_pred (model output): {V_pred.shape} | {V_pred.dtype}")

                V_pred = V_pred.permute(0, 2, 3, 1)
                V_pred = V_pred.squeeze(0)
                V_pred = V_pred[:, :1, :]
                print(f"[LOG] V_pred (after permute): {V_pred.shape} | {V_pred.dtype}")

                V_tr = V_tr.squeeze()
                V_tr = V_tr.T[:20]
                V_tr = V_tr.unsqueeze(-1).permute(0, 2, 1)

                print(f"[LOG] V_tr (squeezed): {V_tr.shape} | {V_tr.dtype}")

                if V_tr.shape[-1] < V_pred.shape[-1]:
                    pad_width = V_pred.shape[-1] - V_tr.shape[-1]
                    V_tr = F.pad(V_tr, (0, pad_width), mode='constant', value=0)  # [20, 1, 5]

                A_tr = A_tr.squeeze()
                print(f"[LOG] A_tr (squeezed): {A_tr.shape} | {A_tr.dtype}")

                # V_pred = V_pred.squeeze()
                # print(f"[LOG] V_pred (final squeeze): {V_pred.shape} | {V_pred.dtype}")
                # if V_tr.ndim == 2 and V_tr.shape[0] == V_pred.shape[-1]:
                #     V_tr = V_tr.transpose(0, 1).unsqueeze(1)  # [2, 30] → [30, 1, 2]

                assert V_pred.shape == V_tr.shape, f"[SHAPE MISMATCH] V_pred: {V_pred.shape}, V_tr: {V_tr.shape}"

                if batch_count % self.batch_size != 0 and cnt != turn_point:
                    l = graph_loss(V_pred, V_tr)
                    if is_fst_loss:
                        loss = l
                        is_fst_loss = False
                    else:
                        loss += l
                else:
                    if batch_count % self.batch_size == 1:
                        continue
                    loss = loss / self.batch_size
                    is_fst_loss = True
                    loss.backward()
                    self.optimizer.step()
                    loss_batch += loss.item()

        self.train_loss.append(loss_batch / batch_count)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            res, DELTA = get_dp_params(self.optimizer)
            print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")

    def find_numbers_above_average(self, numbers):
        average = sum(numbers) / len(numbers)
        threshold = average * 1.5  # 平均数的50%阈值
        outlier_indices = []
        for i, num in enumerate(numbers):
            if num > threshold:
                outlier_indices.append(i)
        return outlier_indices
