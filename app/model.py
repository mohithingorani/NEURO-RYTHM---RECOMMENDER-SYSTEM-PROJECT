import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from app.preprocess import Preprocessor
from sklearn.model_selection import train_test_split


TRACKS = ['t1','t2','t3','t4']


class BioEmbedder(nn.Module):
    def __init__(self, in_dim=7, hid=32):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(in_dim, hid),
        nn.ReLU(),
        nn.Linear(hid, hid),
        nn.ReLU()
)
    def forward(self,x):
        return self.net(x)


class RecommenderModel(nn.Module):
    def __init__(self, bio_dim=32, track_count=4):
        super().__init__()
        self.bio = BioEmbedder(in_dim=7, hid=bio_dim)
        self.head = nn.Linear(bio_dim, track_count)


    def forward(self,x):
        b = self.bio(x)
        scores = self.head(b)
        return scores




def train_model():
    df = pd.read_csv('data/simulated_sessions.csv')
    prep = Preprocessor()
    X, moods, oracles = prep.fit_transform(df)
    y = np.array([TRACKS.index(t) for t in oracles])


    Xtr, Xv, ytr, yv = train_test_split(X,y,test_size=0.2, random_state=42)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xv = torch.tensor(Xv, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    yv = torch.tensor(yv, dtype=torch.long)


    model = RecommenderModel()
    optimiz = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()


    for epoch in range(60):
        model.train()
        optimiz.zero_grad()
        out = model(Xtr)
        loss = loss_fn(out, ytr)
        loss.backward()
        optimiz.step()


    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(Xv).argmax(dim=1)
            acc = (pred==yv).float().mean().item()
            print(f'epoch {epoch} loss={loss.item():.4f} val_acc={acc:.3f}')


# save model and preprocessor state
        torch.save({'model_state': model.state_dict()}, 'data/model.pt')
        joblib.dump(prep, 'data/prep.joblib')
        print('Saved model and preprocessor to data/')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
if args.train:
    train_model()
else:
    print('Run with --train to train the demo model')

