# Day 81 — Logistic Regression visualizer: hosting

Source: Ainain's `streamlit-viz-tool.py`, shared in the ML Guider chat
2026-11-16 — hosted EXACTLY as shared, no code changes.

## Why the pinned versions (not a code change)

His file uses `penalty='none'` (string) and `multi_class` — both removed
from newer scikit-learn, verified by running his exact calls:
- scikit-learn 1.9.1: `multi_class` AND `penalty='none'` both rejected
- 1.6.1 / 1.5.2 / 1.4.2: `multi_class` works, `penalty='none'` rejected
- 1.3.2: both accepted (`penalty='none'` deprecated with a warning,
  confirmed in 1.3.2's own source: `# TODO(1.4): Remove "none" option`)

So the deploy pins scikit-learn 1.3.2 + numpy 1.26.4 (1.3.2 predates
numpy 2.0) on Python 3.11 — deployment config only, the app code is
untouched. Expect a deprecation warning for `penalty='none'`; the app
still runs. Invalid solver/penalty pairs (e.g. `lbfgs` + `l1`) error
exactly as in the original — see the Day 81 notes §3 table.

## Deploy (2 minutes, Ainain's account)

1. This folder is pushed to GitHub as `Video_081/streamlit-app/` in
   `ainainDev/ml-demos` (files: `app.py`, `requirements.txt`,
   `.python-version`).
2. Go to https://share.streamlit.io → New app → pick repo
   `ainainDev/ml-demos`, branch `main`, main file path
   `Video_081/streamlit-app/app.py` → Advanced settings → Python
   version **3.11** → Deploy.
3. The live URL looks like `https://<app-name>.streamlit.app` — paste
   it into the Day 81 notes (§6b placeholder `STREAMLIT_APP_URL`).

## Local run (optional)

```
pip install -r requirements.txt
streamlit run app.py
```
