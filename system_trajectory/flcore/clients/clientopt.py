import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import FedadamOptimizer
# from utils.privacy import *
from utils.trajectory_utils import *


class clientOPT(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        tau = 1e-8
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                         weight_decay=0.0001)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

        # differential privacy
        if self.privacy:
            check_dp(self.model)
            initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

    def train(self, epoch=0):
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
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
            # self.scheduler.step()
            # batch_loss_lst = []
            for cnt, batch in enumerate(trainloader):
                batch_count += 1
                # Get data
                if torch.cuda.is_available():
                    batch = [tensor.cuda() if isinstance(tensor, torch.Tensor) else tensor for tensor in batch]
                else:
                    batch = [tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor for tensor in batch]
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
                loss_mask, V_obs, A_obs, V_tr, A_tr = batch
                self.optimizer.zero_grad(set_to_none=True)
                # Forward
                # V_obs = batch,seq,node,feat
                # V_obs_tmp = batch,feat,seq,node
                V_obs_tmp = V_obs.permute(0, 3, 1, 2)
                V_pred, _ = self.model(V_obs_tmp, A_obs.squeeze())
                V_pred = V_pred.permute(0, 2, 3, 1)
                V_tr = V_tr.squeeze()
                A_tr = A_tr.squeeze()
                V_pred = V_pred.squeeze()
                if batch_count % self.batch_size != 0 and cnt != turn_point:
                    l = graph_loss(V_pred, V_tr)
                    # batch_loss_lst.append(l.item())
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
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    # Metrics
                    loss_batch += loss.item()
                    # print("Averaged Train Loss: {:.4f}".format(loss_batch / batch_count))
                    # print('TRAIN:', '\t LocalEpoch:', step, '\t Loss:', loss_batch / batch_count)
            # if epoch == 5 and step == max_local_steps-1:
            #     self.bad_batch_lst = self.find_numbers_above_average(batch_loss_lst)
            #     print("***")
            #     print(self.bad_batch_lst)
            #     print("***")

        self.train_loss.append(loss_batch / batch_count)
        # if type(x) == type([]):
        #     x[0] = x[0].to(self.device)
        # else:
        #     x = x.to(self.device)
        # y = y.to(self.device)
        # if self.train_slow:
        #     time.sleep(0.1 * np.abs(np.random.rand()))
        # self.optimizer.zero_grad()
        # output = self.model(x)
        # loss = self.loss(output, y)
        # loss.backward()
        # if self.privacy:
        #     dp_step(self.optimizer, i, len(trainloader))
        # else:
        #     self.optimizer.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            res, DELTA = get_dp_params(self.optimizer)
            print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")
