FROM continuumio/miniconda3

WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN conda install python=3.9 -y && \
    pip install -r requirements.txt && \
    conda clean -afy

# Copy the rest of the source code
COPY . .

# Default CMD to run training with reduced rounds (-gr 2)
CMD ["python", "system_trajectory/train.py", "-data", "HIGHD", "-m", "stgcn", "-go", "knative-run", "-algo", "FedAvg", "-nc", "2", "-ls", "1", "-jr", "1", "-lbs", "16", "-gr", "2", "-dev", "cpu", "--n_stgcnn", "4", "--n_txpcnn", "5", "--weight1", "1.0", "--weight2", "0.5"]
