import streamlit as st
import requests
import pandas as pd
import numpy as np


API_URL = 'http://localhost:8000'


st.set_page_config(page_title='NeuroRhythm Demo', layout='centered')


page = st.sidebar.selectbox('Page', ['Live Recommend', 'Simulate Sessions', 'Bandit Monitor'])


if page == 'Live Recommend':
    st.header('Live Recommend')
    st.write('Input current biometric snapshot')
    delta = st.slider('delta', 0.0, 1.0, 0.1)
    theta = st.slider('theta', 0.0, 1.0, 0.1)
    alpha = st.slider('alpha', 0.0, 1.0, 0.3)
    beta = st.slider('beta', 0.0, 1.0, 0.1)
    gamma = st.slider('gamma', 0.0, 1.0, 0.1)
    hr = st.number_input('hr', value=70)
    hrv = st.number_input('hrv', value=50)
    if st.button('Get Recommendation'):
        payload = dict(delta=float(delta), theta=float(theta), alpha=float(alpha), beta=float(beta), gamma=float(gamma), hr=int(hr), hrv=float(hrv))
        try:
            r = requests.post(API_URL + '/recommend', json=payload, timeout=5).json()
            st.json(r)
        except Exception as e:
            st.error('Could not contact API. Run uvicorn app.api:app --reload')


elif page == 'Simulate Sessions':
    st.header('Simulate Sessions & Evaluate')
    st.write('Run batch simulation using local synthetic dataset and call the API to gather recommendations')
    if st.button('Run Simulation (50 samples)'):
        df = pd.read_csv('data/simulated_sessions.csv').sample(50, random_state=1)
        results = []
        for _, row in df.iterrows():
            payload = dict(delta=float(row.delta), theta=float(row.theta), alpha=float(row.alpha), beta=float(row.beta), gamma=float(row.gamma), hr=int(row.hr), hrv=float(row.hrv))
            try:
                r = requests.post(API_URL + '/recommend', json=payload, timeout=3).json()
                rec = r.get('recommended_track')
            except Exception:
                rec = None
            results.append({'session': row.session_id, 'mood': row.mood, 'oracle': row.oracle_track, 'rec': rec})
        resdf = pd.DataFrame(results)
        st.table(resdf.head(20))
        acc = (resdf.oracle == resdf.rec).mean()
        st.metric('Oracle match accuracy', f'{acc:.2f}')


else:
    st.header('Bandit Monitor')
    st.write('This page shows lightweight bandit internals (in-memory)')
    st.write('Note: bandit state is ephemeral and kept in API process memory')
    try:
        r = requests.get(API_URL + '/_bandit_state', timeout=2)
        st.json(r.json())
    except Exception:
        st.info('API does not expose bandit state. Check terminal running the API.')
