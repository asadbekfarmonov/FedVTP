from flask import Flask, request, jsonify
import threading
import torch
import argparse
import sys
import os
import json
from datetime import datetime
import re

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flcore.clients.clientavg import clientAVG
from utils.minio_utils import upload_model, upload_file, download_model
from utils.highd import highdDataset
from flcore.trainmodel.stgcn import social_stgcnn

app = Flask(__name__)

def load_data(dataset_name, site_id):
    root_path = os.path.join(dataset_name, "rawdata")
    train_data = highdDataset(root_path=root_path, site_id=site_id, split="train")
    test_data = highdDataset(root_path=root_path, site_id=site_id, split="val")
    return train_data, test_data

def run_client_logic(client_id, round_i):
    print(f"👤 [Client {client_id}] Starting Round {round_i}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = social_stgcnn(
        n_stgcnn=4,
        n_txpcnn=5,
        output_feat=5,
        seq_len=15,
        kernel_size=7,
        pred_seq_len=25
    ).to(device)

    previous_round = round_i - 1
    global_model_path = f"/tmp/global_model_round_{previous_round}.pth"
    download_model("fedvtp-models", f"global/global_model_round_{previous_round}.pth", global_model_path)
    model.load_state_dict(torch.load(global_model_path, map_location=device))

    train_data, test_data = load_data("highd", client_id)
    client = clientAVG(
        argparse.Namespace(
            batch_size=10,
            local_steps=3,
            local_learning_rate=0.01,
            dataset="highd",
            num_classes=16,
            privacy=False,
            dp_sigma=0.0,
            save_folder_name="models",
            modelname="stgcn",
            device=device,
            model=model
        ),
        id=client_id,
        train_samples=train_data,
        test_samples=test_data,
        train_slow=False,
        send_slow=False
    )

    client.set_parameters(model)
    client.train()

    local_model_path = f"/tmp/client_{client_id}_round_{round_i}.pth"
    torch.save(client.model.state_dict(), local_model_path)
    upload_model(local_model_path, "fedvtp-models", f"clients/round_{round_i}/client_{client_id}_round_{round_i}.pth")
    print(f"📤 [Client {client_id}] Uploaded model for round {round_i}")

    metadata = {
        "client_id": client_id,
        "round": round_i,
        "sample_count": len(getattr(train_data, 'data', train_data)),
        "timestamp": datetime.utcnow().isoformat()
    }

    metadata_path = f"/tmp/client_{client_id}_round_{round_i}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(f"[DEBUG] Uploading metadata to fedvtp-models/clients/round_{round_i}/client_{client_id}_round_{round_i}.json")
    upload_file("fedvtp-models", f"clients/round_{round_i}/client_{client_id}_round_{round_i}.json", metadata_path)

    print(f"📤 [Client {client_id}] Uploaded model and metadata for round {round_i}")

@app.route("/health", methods=["GET"])
def health():
    print("🚀🚀🚀 NEW VERSION DEPLOYED 🚀🚀🚀")
    return "OK", 200

@app.route("/start", methods=["POST"])
def start_client():
    try:
        data = request.get_json(force=True)
        print("[EVENT] Received POST to /start")
        print(json.dumps(data, indent=2))

        # If CloudEvent wrapping is used
        if "data" in data:
            data = data["data"]

        object_key = None
        round_i = None

        # MinIO-style format
        if "Records" in data:
            object_key = data["Records"][0]["s3"]["object"]["key"]
        elif "Key" in data:
            object_key = data["Key"]

        # Check for explicit round number
        if "round" in data:
            round_i = int(data["round"])
            print(f"[INFO] Using round number from payload: {round_i}")
        elif object_key:
            match = re.match(r"global/global_model_round_(\d+)\.pth", object_key)
            if match:
                round_i = int(match.group(1))
                print(f"[INFO] Parsed round number from object key: {round_i}")

        if object_key is None:
            return jsonify({"status": "ignored", "reason": "missing object key"}), 400

        if round_i is None:
            return jsonify({"status": "ignored", "reason": "could not determine round"}), 400

        client_id = os.environ.get("CLIENT_ID", "0")
        thread = threading.Thread(target=run_client_logic, args=(client_id, round_i))
        thread.start()

        return jsonify({"status": "started", "client_id": client_id, "round": round_i}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
