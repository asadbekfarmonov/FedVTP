#run_federated_rounds.py
import os
import requests
import time
import subprocess
import sys

NUM_CLIENTS = 2
MAX_ROUNDS = 5
WAIT_INTERVAL = 10  # seconds
MAX_WAIT_TIME = 600  # seconds max to wait per round

print("🚀 Starting federated rounds job...", flush=True)

MC_ALIAS = "local"
BUCKET = "fedvtp-models"
MINIO_URL = "http://minio.default.svc.cluster.local:9000"

CLIENT_ENDPOINTS = [
    "http://fedvtp-client-0.default.svc.cluster.local/start",
    "http://fedvtp-client-1.default.svc.cluster.local/start",
]
SERVER_AGGREGATE_URL = "http://fedvtp-server.default.svc.cluster.local/aggregate"


def setup_mc_alias():
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    print(f"[DEBUG] MINIO_ACCESS_KEY: {access_key}", flush=True)
    print(f"[DEBUG] MINIO_SECRET_KEY: {secret_key}", flush=True)
    
    if not access_key or not secret_key:
        print("[ERROR] Missing MinIO credentials. Exiting early.", flush=True)
        return False

    try:
        result = subprocess.run([
            "mc", "alias", "set", MC_ALIAS, MINIO_URL, access_key, secret_key
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[INFO] mc alias configured successfully.", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to set mc alias: {e.stderr.decode().strip()}", flush=True)
        return False



def check_all_clients_uploaded(round_i):
    for client_id in range(NUM_CLIENTS):
        key = f"clients/round_{round_i}/client_{client_id}_round_{round_i}.pth"
        try:
            result = subprocess.run(["mc", "stat", f"{MC_ALIAS}/{BUCKET}/{key}"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                return False
        except Exception as e:
            print(f"[ERROR] Checking upload for client {client_id}, round {round_i}: {e}")
            return False
    return True


def wait_for_clients(round_i, max_wait=MAX_WAIT_TIME):
    print(f"⏳ Waiting for clients to upload round {round_i} models...")
    waited = 0
    while not check_all_clients_uploaded(round_i):
        if waited >= max_wait:
            print(f"[ERROR] Timeout: clients didn’t upload round {round_i} in {max_wait}s.")
            return False
        time.sleep(WAIT_INTERVAL)
        waited += WAIT_INTERVAL
    print(f"✅ All clients uploaded for round {round_i}")
    return True


def trigger_clients(global_model_round, target_round):
    for endpoint in CLIENT_ENDPOINTS:
        payload = {
            "Key": f"global/global_model_round_{global_model_round}.pth",
            "round": target_round
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=10)
            print(f"[CLIENT] Triggered {endpoint}: {response.status_code} {response.text}")
        except Exception as e:
            print(f"[ERROR] Could not reach client endpoint {endpoint}: {e}")


def trigger_aggregation():
    try:
        response = requests.post(SERVER_AGGREGATE_URL, timeout=300)
        print(f"[SERVER] Aggregation response: {response.status_code} {response.text}")
    except Exception as e:
        print(f"[ERROR] Could not reach aggregation endpoint: {e}")


if __name__ == "__main__":
    print("🔧 Setting up MinIO alias...")
    if not setup_mc_alias():
        print("❌ Failed to configure MinIO. Exiting.", flush=True)
        sys.exit(1)

    for current_round in range(MAX_ROUNDS):
        print(f"\n🚀 ROUND {current_round + 1} STARTING")
        trigger_clients(current_round, current_round + 1)

        if not wait_for_clients(current_round + 1):
            print(f"⚠️ Skipping aggregation for round {current_round + 1} due to missing client uploads.")
            continue

        trigger_aggregation()

    print("\n🎉 Federated training rounds completed.")
