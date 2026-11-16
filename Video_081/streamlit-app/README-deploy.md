# Day 81 — Logistic Regression visualizer: hosting

Source: Ainain's `streamlit-viz-tool.py`, shared in the ML Guider chat
2026-11-16 02:26 UTC, fixed for current scikit-learn (see `app.py`
header for the 6 fixes — the original crashes on scikit-learn >= 1.7).

## Why Streamlit Cloud (not GitHub Pages)

GitHub Pages (where the 112 static demos live) serves static files only —
it cannot run Python, so a Streamlit app cannot live there. The free
host for this is Streamlit Community Cloud (share.streamlit.io).

## Deploy (2 minutes, Ainain's account)

1. Push this folder to GitHub as `Video_081/streamlit-app/` in
   `ainainDev/ml-demos` (files: `app.py`, `requirements.txt`) —
   same `github push_files` route used for the demo migration.
2. Go to https://share.streamlit.io → New app → pick repo
   `ainainDev/ml-demos`, branch `main`, main file path
   `Video_081/streamlit-app/app.py` → Deploy.
3. The live URL looks like
   `https://<app-name>.streamlit.app` — paste it into the Day 81 notes
   (section added below, placeholder `STREAMLIT_APP_URL`) and into the
   ML Guider app, then verify: Binary + Run → boundary + accuracy shows;
   switch penalty to `l1` + solver `saga` → runs; `lbfgs` + `l1` →
   friendly error, not a crash.

## Local run (optional)

```
pip install -r requirements.txt
streamlit run app.py
```
