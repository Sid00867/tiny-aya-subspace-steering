"""Upload exp6_overrefusal_compliance artifact bundle to Hugging Face Hub for immediate reviewer access."""
from huggingface_hub import HfApi
import sys
repo_id = sys.argv[1] if len(sys.argv) > 1 else "YOUR_HF_USERNAME/exp6_overrefusal_compliance_artifacts"
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
api.upload_folder(folder_path="lexsi_hackathon_artifacts/exp6_overrefusal_compliance", repo_id=repo_id, repo_type="dataset")
print("✅ Uploaded bundle to https://huggingface.co/datasets/" + repo_id)
