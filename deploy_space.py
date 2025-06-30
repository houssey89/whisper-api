import os
import shutil
from huggingface_hub import HfApi, Repository


HF_PAT = os.getenv("HF_PAT")
if not HF_PAT:
    raise RuntimeError("HF_PAT environment variable not set")

SPACE_REPO = "Houssey/whisper-jules"
REPO_URL = f"https://huggingface.co/spaces/{SPACE_REPO}"
LOCAL_DIR = "whisper-jules"

FILES = ["server.py", "requirements.txt", "Dockerfile", ".huggingface.yaml"]

def main():
    # Clone the space repository
    repo = Repository(local_dir=LOCAL_DIR, clone_from=REPO_URL, token=HF_PAT)

    # Copy backend files into the repo
    for file in FILES:
        shutil.copy(file, os.path.join(LOCAL_DIR, file))

    # Stage, commit and push
    repo.git_add(pattern=".")
    repo.git_commit("init backend files")
    repo.git_push()

    # Add secrets via API
    api = HfApi(token=HF_PAT)
    api.add_space_secret(repo_id=SPACE_REPO, key="OPENAI_API_KEY", value="XXX")
    api.add_space_secret(repo_id=SPACE_REPO, key="JULES_API_URL", value="https://api.lovable.so/functions/jules")
    api.add_space_secret(repo_id=SPACE_REPO, key="MED_MODEL_ID", value="housseynatou/clip-meds")

if __name__ == "__main__":
    main()
