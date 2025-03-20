from huggingface_hub import snapshot_download
import os

# Create a data directory in the project folder
data_dir = os.path.join(os.path.dirname(__file__), "download2")
os.makedirs(data_dir, exist_ok=True)

print(f"Downloading dataset to: {data_dir}")
snapshot_download(repo_id="LMUK-RADONC-PHYS-RES/TrackRAD2025", 
                 repo_type="dataset", 
                 local_dir=data_dir)


