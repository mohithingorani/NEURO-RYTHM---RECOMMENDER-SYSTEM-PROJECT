from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch
import numpy as np
from app.model import RecommenderModel, TRACKS
from app.recommender import LinUCB


app = FastAPI()


# load model and preprocessor
try:
    state = torch.load('data/model.pt', map_location='cpu')
    model = RecommenderModel()
    model.load_state_dict(state['model_state'])
    model.eval()
    prep = joblib.load('data/prep.joblib')
except Exception as e:
    print('Model not found; run training first.', e)
    model=None
    prep=None


linucb = LinUCB(n_arms=len(TRACKS), dim=32, alpha=0.5)


class SessionInput(BaseModel):
    delta: float
    theta: float
    alpha: float
    beta: float
    gamma: float
    hr: int
    hrv: float


@app.post('/recommend')
def recommend(sess: SessionInput):
    if model is None:
        return {'error':'model not trained yet'}
    import numpy as np
    arr = np.array([[sess.delta,sess.theta,sess.alpha,sess.beta,sess.gamma,sess.hr,sess.hrv]])
    X = prep.transform(pd.DataFrame(arr, columns=['delta','theta','alpha','beta','gamma','hr','hrv']))
    Xtensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        scores = model(Xtensor).numpy().reshape(-1)
    # Use scores as feature to LinUCB: simple projection via model.bio output (deterministic approach omitted for speed)
    # For demo, create a 32-d context by forwarding through bio submodule
    import torch
    bio = model.bio
    ctx = bio(torch.tensor(X, dtype=torch.float32)).numpy().reshape(-1)
    arm, p = linucb.select(ctx)
    return {'recommended_track': TRACKS[arm], 'scores': scores.tolist(), 'linucb_p': p}


class RewardInput(BaseModel):
    track_id: str
    reward: float # 0..1
    ctx: list


@app.post('/reward')
def reward(inp: RewardInput):
    try:
        arm = TRACKS.index(inp.track_id)
    except ValueError:
        return {'error':'unknown track'}
    x = np.array(inp.ctx)
    linucb.update(arm, x, float(inp.reward))
    return {'status':'updated'}