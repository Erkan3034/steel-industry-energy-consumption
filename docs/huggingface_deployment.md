# Hugging Face Spaces Deployment

The Space runs the Streamlit app in a Docker container (`sdk: docker`, port 8501). Only the runtime files are deployed: `app/`, `src/`, `models/`, `requirements.txt`, `Dockerfile`, and `hf_space/README.md` (renamed to `README.md` — it holds the Space config front matter).

## 1. Create the Space

1. https://huggingface.co/new-space
2. **Owner:** your account · **Space name:** `steel-energy-predictor`
3. **SDK:** Docker → **Blank** template · **Hardware:** CPU basic (free) · **Public**
4. Create.

## 2. Create an access token

https://huggingface.co/settings/tokens → **New token** → type **Write** (or fine-grained with write access to this Space). Keep it private.

## Option A — Automatic sync from GitHub (recommended)

`.github/workflows/deploy-hf.yml` uploads the Space files on every push to `main` that touches runtime code.

1. GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**
2. Name: `HF_TOKEN`, value: the token from step 2.
3. If your Space id differs, edit `HF_SPACE` in the workflow.
4. **Actions → Deploy to Hugging Face Space → Run workflow** (or just push to `main`).

The upload goes through the Hub API, which handles the `.pkl` binary automatically.

## Option B — Manual push with git

```bash
pip install -U huggingface_hub
hf auth login                      # paste the write token
git clone https://huggingface.co/spaces/erkanturgut/steel-energy-predictor hf-space
```

Copy the runtime files into the clone:

```powershell
# PowerShell, from the project root
Copy-Item -Recurse -Force app, src, models, requirements.txt, Dockerfile, LICENSE, MODEL_CARD.md hf-space\
Copy-Item -Force hf_space\README.md hf-space\README.md
```

Commit and push (binaries must go through LFS/Xet):

```bash
cd hf-space
git lfs install
git lfs track "*.pkl"
git add .
git commit -m "Deploy Streamlit app"
git push
```

## 3. Verify

- Space page → **Logs** tab: build should end with `You can now view your Streamlit app`.
- Common failures:
  - `AttributeError: Can't get attribute '_RemainderColsList'` / `InconsistentVersionWarning` → scikit-learn version differs from the one the model was saved with. Keep `scikit-learn==1.6.1` or retrain and re-pin.
  - App stuck on "Starting" → `app_port` in the README front matter must match `--server.port` (8501).
  - `ModuleNotFoundError: src` → `src/` was not uploaded.

## 4. Local container test

```bash
docker build -t steel-energy .
docker run --rm -p 8501:8501 steel-energy
# http://localhost:8501
```
