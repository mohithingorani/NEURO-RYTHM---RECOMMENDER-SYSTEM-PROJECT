This repository is a **part of a Recommender Systems Project** — focusing on personalized, emotion-aware music therapy through physiological signal analysis.


It contains a prototype implementation of **NeuroRhythm**, a mental-state-aware music therapy recommender that uses simulated physiological signals (EEG-like + heart rate), text mood labels, and audio metadata to recommend personalized music/soundscapes.


This codebase is a compact, runnable prototype intended for research/portfolio use. It includes:
- data simulator (synthetic EEG/HR + mood labels)
- preprocessing pipeline
- a lightweight multimodal model (PyTorch) that fuses biometric embeddings and context
- a contextual-bandit style recommender for online personalization (LinUCB)
- FastAPI REST endpoint and a Streamlit multi-page app for demo


## Quick start
1. Create a virtualenv: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Generate synthetic data: `python -m app.data_simulator`
4. Train model (demo): `python -m app.model --train`
5. Run API: `uvicorn app.api:app --reload --port 8000`
6. Run UI: `streamlit run app/ui_streamlit.py