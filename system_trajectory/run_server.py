import os, sys, time, torch, json, copy, traceback
from flask import Flask, request, jsonify
from flcore.servers.serveravg_pure import FedAvg
from utils.minio_utils import upload_model, list_models, download_model, ensure_bucket_exists

# 🔧 Fix: restore path append
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = Flask(__name__)
ensure_bucket_exists("fedvtp-models")

# ✅ Full argument parser restored
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-go', "--goal", type=str, default="test")
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="highd")
    parser.add_argument('-nb', "--num_classes", type=int, default=16)
    parser.add_argument('-m', "--model", type=str, default="stgcn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01)
    parser.add_argument('-gr', "--global_rounds", type=int, default=5)
    parser.add_argument('-ls', "--local_steps", type=int, default=3)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0)
    parser.add_argument('-nc', "--num_clients", type=int, default=4)
    parser.add_argument('-pv', "--prev", type=int, default=0)
    parser.add_argument('-t', "--times", type=int, default=1)
    parser.add_argument('-eg', "--eval_gap", type=int, default=1)
    parser.add_argument('-dp', "--privacy", type=bool, default=False)
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='models')
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0)
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0)
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0)
    parser.add_argument('-ts', "--time_select", type=bool, default=False)
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000)
    parser.add_argument('--n_stgcnn', type=int, default=4)
    parser.add_argument('--n_txpcnn', type=int, default=5)
    parser.add_argument('--weight1', type=float, default=1.0)
    parser.add_argument('--weight2', type=float, default=0.5)
    parser.add_argument('--modelname', type=str, default='stgcn')
    parser.add_argument('--flag', type=str, default='default_run')
    return parser.parse_args([])  # Empty list for script-style run

args = get_args()

# ✅ Device check
if torch.backends.mps.is_available():
    args.device = torch.device("mps")
elif torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

# ✅ Load model with full config
from flcore.trainmodel.stgcn import social_stgcnn
args.model = social_stgcnn(
    n_stgcnn=args.n_stgcnn,
    n_txpcnn=args.n_txpcnn,
    output_feat=5,
    seq_len=15,
    kernel_size=7,
    pred_seq_len=25
).to(args.device)

# ✅ Instantiate FedAvg server
server = FedAvg(args, times=0)
server.current_round = 1  # <- start from round 1

@app.route("/aggregate", methods=["POST"])
def aggregate_round():
    try:
        # Ignore event body completely.
        # Server tracks internally which round it is.
        
        round_i = server.current_round  # <--- SERVER tracks its own current round
        print(f"\n🌐 [SERVER] Aggregating for Round {round_i}...")

        # Save and upload the current global model
        global_model_path = f"/tmp/global_model_round_{round_i}.pth"
        torch.save(server.global_model.state_dict(), global_model_path)
        upload_model(global_model_path, "fedvtp-models", f"global/global_model_round_{round_i}.pth")
        print(f"✅ [SERVER] Uploaded global model for round {round_i}")

        expected_clients = server.join_clients
        client_models = []
        timeout = 300
        start_time = time.time()

        print(f"⏳ [SERVER] Waiting for {expected_clients} client models...")
        while len(client_models) < expected_clients:
            models = list_models("fedvtp-models", prefix=f"clients/round_{round_i}/")
            if len(models) >= expected_clients:
                client_models = models
                break
            if time.time() - start_time > timeout:
                print(f"⚠️ [SERVER] Timeout reached while waiting for client uploads.")
                return jsonify({"status": "timeout", "received": len(client_models)}), 408
            time.sleep(5)

        print(f"⬇️ [SERVER] Downloading client models...")

        # Clear previous round uploads
        server.uploaded_models = []
        server.uploaded_weights = []
        server.uploaded_ids = []

        for model_key in client_models:
            if not model_key.endswith(".pth"):
                continue

            json_key = model_key.replace(".pth", ".json")
            json_path = f"/tmp/{os.path.basename(json_key)}"
            try:
                download_model("fedvtp-models", json_key, json_path)
                with open(json_path) as f:
                    meta = json.load(f)
                weight = meta.get("sample_count", 1)
                client_id = meta.get("client_id", "unknown")

                model_path = f"/tmp/{os.path.basename(model_key)}"
                download_model("fedvtp-models", model_key, model_path)

                model = copy.deepcopy(server.global_model)
                model.load_state_dict(torch.load(model_path, map_location=args.device))

                server.uploaded_models.append(model)
                server.uploaded_weights.append(weight)
                server.uploaded_ids.append(client_id)

            except Exception as e:
                print(f"⚠️ [SERVER] Error handling {model_key}: {e}")
                continue

        if server.uploaded_weights:
            total = sum(server.uploaded_weights)
            server.uploaded_weights = [w / total for w in server.uploaded_weights]
            server.aggregate_parameters()
            print(f"🤝 [SERVER] Aggregated updates for round {round_i}")

            server.current_round += 1  # <--- Increment internal round counter

            return jsonify({"status": "aggregated", "round": round_i}), 200
        else:
            return jsonify({"status": "no client models"}), 400

    except Exception as e:
        print(f"💥 [SERVER] Exception occurred: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    print("🚀🚀🚀 NEW VERSION DEPLOYED 🚀🚀🚀")
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
