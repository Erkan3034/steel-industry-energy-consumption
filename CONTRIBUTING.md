# Contributing

Thanks for your interest! Bug reports, feature ideas and pull requests are welcome.

## Workflow

1. Fork the repo and create a branch from `main`:
   ```bash
   git checkout -b feature/short-description
   ```
2. Set up the environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```
3. Make your changes, then check them:
   ```bash
   ruff check .
   pytest -v
   ```
4. Commit with a clear message ([Conventional Commits](https://www.conventionalcommits.org/) preferred: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
5. Open a pull request describing **what** changed and **why**. CI must pass.

## Guidelines

- Keep feature logic in `src/`; the notebook is for exploration only.
- Any change to features or hyperparameters goes through `src/config.py`.
- If you retrain the model, commit the new `models/*.pkl` **and** `models/metrics.json`, update the metrics in `README.md` and `MODEL_CARD.md`, and keep `scikit-learn` pinned to the version used for training.
- Never commit raw data, credentials or tokens (`HF_TOKEN`, `.env`).
- Add or update tests for behavior changes.

## Reporting Issues

Include: steps to reproduce, expected vs. actual behavior, Python and package versions (`pip freeze`), and the full traceback.
