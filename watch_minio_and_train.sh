#!/bin/bash

BUCKET="fedvtp-models"
CLIENT_ID=0  # set dynamically later via ENV
DATASET="highd"
CHECK_INTERVAL=10  # seconds
SEEN_ROUNDS=()

echo "[👀 WATCHER] Starting MinIO polling for client $CLIENT_ID..."

while true; do
    # List all global models in MinIO
    GLOBAL_MODELS=$(mc ls local/$BUCKET/global/ | grep "global_model_round_" | awk '{print $NF}' | sort)

    for file in $GLOBAL_MODELS; do
        ROUND=$(echo $file | grep -oP 'round_\K[0-9]+')

        # Skip if already processed
        if [[ " ${SEEN_ROUNDS[@]} " =~ " ${ROUND} " ]]; then
            continue
        fi

        echo "[🚀 CLIENT] Triggering training for round $ROUND"
        python run_client.py --client_id $CLIENT_ID --round $ROUND

        SEEN_ROUNDS+=($ROUND)
    done

    sleep $CHECK_INTERVAL
done
