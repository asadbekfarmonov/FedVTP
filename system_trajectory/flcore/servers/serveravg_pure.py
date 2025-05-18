#serveravg_pure.py
import time
import torch
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
import matplotlib.pyplot as plt
import copy
from threading import Thread
import pickle
import os
from statistics import mean
from utils.minio_utils import upload_model


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # Select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)

        self.epoch = 0
        self.dataset = args.dataset
        self.Budget = []

        total_sample = sum(len(client.train_samples) for client in self.clients)
        if args.weight1 == 5 and args.weight2 == 5:
            for client in self.clients:
                self.clients_weight.append(len(client.train_samples) / total_sample)
        else:
            self.clients_weight.append(args.weight1)
            self.clients_weight.append(args.weight2)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        earlystopping = False
        self.selected_clients = self.clients

        for i in range(self.global_rounds + 1):
            print(f"\n{'=' * 10} [ROUND {i}] START {'=' * 10}")
            s_t = time.time()

            # Send global model to clients
            print(f"[LOG] Sending global model to all clients.")
            self.send_models()

            print(f"[LOG] Running initial evaluation before local training...")
            next_state_a, _, loss, rmse, earlystopping = self.evaluate()

            # Train clients
            for j, client in enumerate(self.selected_clients):
                print(f"\n[LOG] Client {client.id} starts training.")
                client.train(self.epoch)

            print(f"\n------------- Round {i} -------------")
            print("\n[LOG] Evaluate personal model after training.")
            next_state_a, _, loss, rmse, earlystopping = self.evaluate()

            # Save client state info
            filtered_state = [0.1 if value > 0.1 else value for value in next_state_a]
            self.states_lst.append(filtered_state)
            print(f"[LOG] Client state values (clipped): {filtered_state}")

            # Collect uploaded models
            print("[LOG] Receiving client models...")
            self.receive_models()

            # Aggregate weights
            print("[LOG] Aggregating models into global model.")
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print(f"[ROUND {i}] Duration: {self.Budget[-1]:.2f}s")

            self.epoch = i
            if earlystopping:
                print("[LOG] Early stopping triggered.")
                break

        print("\n=== Training Completed ===")
        avg_time = sum(self.Budget[1:]) / len(self.Budget[1:]) if len(self.Budget) > 1 else 0
        print(f"\n[LOG] Average time per round: {avg_time:.2f}s")

        # Best epoch and metrics
        min_value = min(self.val_rmse)
        min_index = self.val_rmse.index(min_value)
        print(f"\n[LOG] Best Epoch: {min_index / 2:.1f}")
        print(f"[LOG] Best Val RMSE: {min_value:.4f}")
        print(f"[LOG] Corresponding Test RMSE: {self.test_rmse[min_index]:.4f}")
        print(f"[LOG] Test ADE: {self.test_mae[min_index]:.4f}")
        print(f"[LOG] Test FDE: {self.test_mape[min_index]:.4f}")

        print("\n[LOG] Best client metrics:")
        best_rmse, best_mae, best_mape, weight = [], [], [], []
        for i, client in enumerate(self.clients):
            weight.append(len(client.train_samples))
            min_val_rmse = min(client.val_rmse)
            min_index = client.val_rmse.index(min_val_rmse)

            print(f" - Client {i}:")
            print(f"   - Val RMSE: {min_val_rmse:.4f}")
            print(f"   - Epoch: {min_index}")
            print(f"   - Test RMSE: {client.test_rmse[min_index]:.4f}")
            print(f"   - Test ADE: {client.test_mae[min_index]:.4f}")
            print(f"   - Test FDE: {client.test_mape[min_index]:.4f}")

            best_rmse.append(client.test_rmse[min_index])
            best_mae.append(client.test_mae[min_index])
            best_mape.append(client.test_mape[min_index])

        print("\n[LOG] Aggregated Best Metrics (Weighted):")
        print(" - RMSE:", sum([x * y for x, y in zip(best_rmse, weight)]) / sum(weight))
        print(" - ADE :", sum([x * y for x, y in zip(best_mae, weight)]) / sum(weight))
        print(" - FDE :", sum([x * y for x, y in zip(best_mape, weight)]) / sum(weight))

        # Plot and save visualizations
        for i, client in enumerate(self.clients):
            fig, bx = plt.subplots()
            bx.plot(client.test_rmse, label='test RMSE')
            bx.set_title(f"Client {i} Test RMSE vs Epoch")
            bx.set_xlabel('Epoch')
            bx.set_ylabel('RMSE')
            bx.legend()
            self.save_figure(fig, 'client_rmse', i)

        for i, client in enumerate(self.clients):
            fig, bx = plt.subplots()
            bx.plot(client.val_loss, label='validation loss')
            bx.set_title(f"Client {i} Val Loss vs Epoch")
            bx.set_xlabel('Epoch')
            bx.set_ylabel('Loss')
            bx.legend()
            self.save_figure(fig, 'client_loss', i)

        # Server-level RMSE plot
        fig, bx = plt.subplots()
        bx.plot(self.test_rmse, label='Server Test RMSE', color='g')
        bx.set_title("Server Test RMSE vs Epoch")
        bx.set_xlabel('Epoch')
        bx.set_ylabel('RMSE')
        bx.legend()
        self.save_figure(fig, 'server_rmse', 0)

        # Plot client states over time
        color_box = ['r', 'b', 'g', 'c', 'y', 'k']
        fig, ax = plt.subplots()
        for i in range(self.num_clients):
            state1_lst = [row[i] for row in self.states_lst]
            ax.plot(state1_lst, color=color_box[i], label=f'Client {i}')
        ax.set_title("Client Loss State vs Epoch")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        self.save_figure(fig, 'loss', -1)

        # Plot RMSE comparisons
        rmse_lst = [client.test_rmse for client in self.clients]
        fig, ax = plt.subplots()
        for i in range(self.num_clients):
            ax.plot(rmse_lst[i], color=color_box[i], label=f'Client {i}')
        ax.plot(self.val_rmse, 'g', label='Server')
        ax.set_title('RMSE vs Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.legend()
        self.save_figure(fig, 'rmse', -1)

        # Save output
        save_data = {
            'Best metrics': {
                'Val RMSE': min_value,
                'Epochs': min_index,
                'Test RMSE': self.test_rmse[min_index],
                'Test MAE': self.test_mae[min_index],
                'Test MAPE': self.test_mape[min_index]
            }
        }
        save_data2 = [{'client.val_rmse': client.val_rmse, 'client.val_loss': client.val_loss}
                      for client in self.clients]
        save_data3 = {'val_rmse': self.val_rmse}

        result_path = os.path.join(self.model_path, 'output.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(save_data, f)
            pickle.dump(save_data2, f)
            pickle.dump(save_data3, f)

        # Upload final model to MinIO
        final_model_path = os.path.join(self.model_path, self.algorithm + "_bestmodel.pth")
        if os.path.exists(final_model_path):
            try:
                upload_model(
                    file_path=final_model_path,
                    bucket_name='fedvtp-models',
                    object_name='global_model_latest.pth'
                )
                print("[MinIO] ✅ Uploaded global model to MinIO: global_model_latest.pth")
            except Exception as e:
                print(f"[MinIO] ❌ Failed to upload model to MinIO: {e}")
        else:
            print(f"[MinIO] ⚠️ Model file not found: {final_model_path}")

        return self.epoch
