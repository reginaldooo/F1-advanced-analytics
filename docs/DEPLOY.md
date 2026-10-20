# Deploying the Streamlit app (Community Cloud)

The app only needs the committed table `app/data/drivers.csv`, no Kaggle access.

1. Push `main` to GitHub (`reginaldooo/F1-advanced-analytics`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. New app, then:
   - Repository: `reginaldooo/F1-advanced-analytics`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - Advanced settings: Python 3.12 (3.9+ works).
4. Streamlit installs `app/requirements.txt` (streamlit, pandas, plotly, all pinned). Deploy.
5. Copy the app URL into the README and the LinkedIn post.

After changing the analysis, run `make app-data`, commit `app/data/drivers.csv` and push; the app redeploys automatically.
