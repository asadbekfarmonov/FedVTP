from datetime import datetime

import torch
import os
import numpy as np
import h5py
import copy
import math
import random

from torch.utils.data import DataLoader

from utils.early_stopping import EarlyStopping
import dill

from utils.data_utils import read_client_data
# from utils.ngsim import ngsimDataset
from utils.highd import highdDataset


class Server(object):
    def __init__(self, args, times):
        # torch.manual_seed(0)
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.original_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100

        # my diy
        # self.model = copy.deepcopy(args.model)
        model_path = os.path.join("models", self.dataset)
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        model_string = args.algorithm + "_" + args.dataset
        print(f"{date_string}")
        model_path = os.path.join(model_path, f"model_{model_string}_{date_string}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model_path = model_path
        model_path = os.path.join(model_path, self.algorithm + "_bestmodel" + ".pth")
        self.early_stopping = EarlyStopping(path=model_path, id=-1)
        self.prev_rmse = 0

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_rmse = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        # 2.初始化客户端权重
        self.clients_weight = []
        self.rewards_lst = []
        self.actions_lst = []
        self.states_lst = []
        self.last_state = []
        self.val_rmse = []
        self.test_rmse = []
        self.test_mae = []
        self.test_mape = []
        self.p_val_rmse = []
        self.p_test_rmse = []
        self.p_test_mae = []
        self.p_test_mape = []
        self.val_batch_weight = []
        self.epoch = 0
        if args.modelname == 'stgcn':
            self.graph = True
        else:
            self.graph = False

    def load_val_data_from_csv(self, dataset_name, site_id):
        print(f"Loading validation data for {dataset_name}, site {site_id}")
        root_path = os.path.join(dataset_name, "rawdata")
        dataset = highdDataset(root_path=root_path, site_id=site_id, split="val")
        return dataset

    def load_data_from_csv(self, dataset_name, site_id):
        print(f"Loading raw data for {dataset_name}, site {site_id}")
        root_path = os.path.join(dataset_name, "rawdata")
        dataset = highdDataset(root_path=root_path, site_id=site_id, split="train")
        return dataset

    def set_clients(self, args, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            model_path = os.path.join("models", args.dataset)
            # train_data = read_client_data(self.dataset, i, is_train=True)
            # test_data = read_client_data(self.dataset, i, is_train=False)
            #
            # with open(model_path + '/' + str(i) + '_' + args.dataset + 'noise' + '_train_data.pkl', 'wb') as f:
            #     dill.dump(train_data, f)
            # with open(model_path + '/' + str(i) + '_' + args.dataset + 'noise' + '_test_data.pkl', 'wb') as f:
            #     dill.dump(test_data, f)
            try:
                with open(model_path + '/' + str(i) + '_' + args.dataset + '_train_data.pkl', 'rb') as f:
                    train_data = pickle.load(f)
            except FileNotFoundError:
                print(f"[INFO] Preprocessed data not found. Loading raw CSVs for site {i} ...")
                train_data = self.load_data_from_csv(args.dataset, i)  # Replace with your actual loading logic

            try:
                with open(model_path + '/' + str(i) + '_' + args.dataset + '_test_data.pkl', 'rb') as f:
                    val_data = pickle.load(f)
            except FileNotFoundError:
                print(f"[INFO] Preprocessed test data not found. Loading raw val CSVs for site {i} ...")
                val_data = self.load_val_data_from_csv(args.dataset, i)

                
            print("🧠 THIS IS THE NEW VERSION")
            print(f"[DEBUG] len(train_data): {len(train_data)}")

            print(f"[DEBUG] len(train_samples): {len(train_data)}")
            print(f"[DEBUG] len(test_samples): {len(val_data)}")


            client = clientObj(args,
                               id=i,
                               train_samples=train_data,
                               test_samples=val_data,
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):

        selected_clients = list(np.random.choice(self.clients, self.join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert len(self.clients) > 0
        print(f"[Server] Sending global model to {len(self.clients)} clients.")
        for client in self.clients:
            client.set_parameters(self.global_model)

    def receive_models(self):
        assert len(self.selected_clients) > 0
        print(f"[Server] Receiving models from {len(self.selected_clients)} clients...")

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []

        for i, client in enumerate(self.selected_clients):
            print(f"[Client {client.id}] Model received.")
            self.uploaded_weights.append(self.clients_weight[i])
            tot_samples += self.uploaded_weights[i]
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        print("[Server] Normalized weights:")
        for cid, w in zip(self.uploaded_ids, self.uploaded_weights):
            print(f" - Client {cid}: weight = {w:.4f}")

    def receive_models(self):
        assert len(self.selected_clients) > 0
        print(f"[Server] Receiving models from {len(self.selected_clients)} clients...")

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []

        for i, client in enumerate(self.selected_clients):
            print(f"[Client {client.id}] Model received.")
            self.uploaded_weights.append(self.clients_weight[i])
            tot_samples += self.uploaded_weights[i]
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        print("[Server] Normalized weights:")
        for cid, w in zip(self.uploaded_ids, self.uploaded_weights):
            print(f" - Client {cid}: weight = {w:.4f}")

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0
        print("[Server] Aggregating parameters...")

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
        print("[Server] Aggregation complete.")

    def add_parameters(self, w, client_model):
        print(f"[Server] Adding parameters with weight {w:.4f}")
        for i, (server_param, client_param) in enumerate(
                zip(self.global_model.parameters(), client_model.parameters())):
            if server_param.shape != client_param.shape:
                print(f"[WARN] Param shape mismatch at layer {i}: "
                      f"server={server_param.shape}, client={client_param.shape}")
            server_param.data += client_param.data.clone() * w

    def set_parameters(self, new_model, old_model):
        for new_param, old_param in zip(new_model.parameters(), old_model.parameters()):
            old_param.data = new_param.data.clone()

    def save_global_model(self):
        model_path = os.path.join(self.model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def save_figure(self,fig, name, i):
        model_path = os.path.join(self.model_path, name + str(i) + '.png')
        fig.savefig(model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_rmse)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_rmse', data=self.rs_test_rmse)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        num_samples = []
        losses = []
        tot_rmse = []
        tot_mae = []
        tot_mape = []
        tot_batch_loss = []
        tot_val_rmse = []
        for c in self.clients:
            test_num, test_loss, test_rmse, test_mae, test_mape, val_rmse, stopped = c.test_metrics()
            # test_rmse = test_rmse.item()
            tot_rmse.append(test_rmse)
            tot_mae.append(test_mae)
            tot_mape.append(test_mape)
            num_samples.append(test_num)
            tot_val_rmse.append(val_rmse)
            losses.append(test_loss * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_rmse, tot_mae, tot_mape, losses, tot_val_rmse


    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            losses.append(ns)
            num_samples.append(cl)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, rmse=None, es=True):
        print("[Server] Starting evaluation...")
        stats = self.test_metrics()
        test_rmse, test_mae, test_mape, val_rmse, test_loss = 0, 0, 0, 0, 0

        for a, n in zip(stats[2], stats[1]):
            test_rmse += a * n
        for a, n in zip(stats[3], stats[1]):
            test_mae += a * n
        for a, n in zip(stats[4], stats[1]):
            test_mape += a * n
        for a, n in zip(stats[6], stats[1]):
            val_rmse += a * n
        for a, n in zip(stats[5], stats[1]):
            test_loss += a * n

        total_samples = sum(stats[1])
        test_rmse /= total_samples
        test_mae /= total_samples
        test_mape /= total_samples
        val_rmse /= total_samples
        test_loss /= total_samples

        print(f"[Eval] Averaged Val RMSE:  {val_rmse:.4f}")
        print(f"[Eval] Averaged Test RMSE: {test_rmse:.4f}")
        print(f"[Eval] Averaged Test MAE:  {test_mae:.4f}")
        print(f"[Eval] Averaged Test MAPE: {test_mape:.4f}")

        self.val_rmse.append(min(val_rmse, 40))
        self.test_rmse.append(min(test_rmse, 40))
        self.test_mae.append(min(test_mae, 30))
        self.test_mape.append(min(test_mape, 50))

        if es:
            self.early_stopping(val_rmse, self.global_model, "rmse")
            if self.early_stopping.early_stop:
                print("[Server] Early stopping triggered!")

        return stats[5], stats[1], test_loss, test_rmse, self.early_stopping.early_stop

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def not_nan(self, lst):
        for item in lst:
            if math.isnan(item):
                return False
        return True
