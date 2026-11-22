# Hugging Face Spaces Deployment

The Space uses the **Gradio SDK** (free tier; the Docker SDK now requires a paid plan). The Streamlit app (`app/app.py`) and the `Dockerfile` are for local / self-hosted use.

Deployed files (staged by `.github/workflows/deploy-hf.yml`):

| Space path | Source |
| :--- | :--- |
| `README.md` | `hf_space/README.md` (Space config front matter: `sdk: gradio`, `app_file: app/gradio_app.py`) |
| `requirements.txt` | `hf_space/requirements.txt` (no Gradio/Streamlit — Gradio comes from `sdk_version`) |
| `app/gradio_app.py` | `app/gradio_app.py` |
| `src/`, `models/` | same |
| `LICENSE`, `MODEL_CARD.md` | same |

## 1. Create the Space

1. https://huggingface.co/new-space
2. **Owner:** `erkanturgut` · **Space name:** `steel-energy-predictor` · **License:** MIT
3. **SDK:** Gradio → **Blank**
4. **Hardware:** the free option (CPU Basic, or ZeroGPU if CPU Basic is unavailable — the model runs on CPU either way)
5. **Public** → Create Space

## 2. Create an access token

https://huggingface.co/settings/tokens → **New token** → type **Write**. Keep it private.

## Option A — Automatic sync from GitHub (recommended)

1. GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**
2. Name: `HF_TOKEN`, value: the token from step 2.
3. **Actions → Deploy to Hugging Face Space → Run workflow** (afterwards it runs on every push to `main` that touches `app/`, `src/`, `models/` or `hf_space/`).

The upload goes through the Hub API, which handles the `.pkl` binary automatically.

## Option B — Manual upload with the HF CLI

```bash
pip install -U huggingface_hub
hf auth login
```

```powershell
# PowerShell, from the project root: stage the Space files
$stage = "hf-stage"
New-Item -ItemType Directory -Force "$stage\app" | Out-Null
Copy-Item -Recurse -Force src, models, LICENSE, MODEL_CARD.md $stage
Copy-Item -Force app\gradio_app.py "$stage\app\"
Copy-Item -Force hf_space\README.md "$stage\README.md"
Copy-Item -Force hf_space\requirements.txt "$stage\requirements.txt"
hf upload erkanturgut/steel-energy-predictor $stage . --repo-type space
```

## 3. Verify

- Space page → **Logs**: build ends with `Running on local URL: http://0.0.0.0:7860`.
- Common failures:
  - `AttributeError: Can't get attribute '_RemainderColsList'` / `InconsistentVersionWarning` → scikit-learn version differs from the one the model was saved with. Keep `scikit-learn==1.6.1` or retrain and re-pin.
  - `ModuleNotFoundError: src` → `src/` was not uploaded.
  - `No application file` → `app_file` in the README front matter must point to `app/gradio_app.py`.

## Local runs

```bash
python app/gradio_app.py          # http://localhost:7860
streamlit run app/app.py          # http://localhost:8501
docker build -t steel-energy . && docker run --rm -p 8501:8501 steel-energy
```
