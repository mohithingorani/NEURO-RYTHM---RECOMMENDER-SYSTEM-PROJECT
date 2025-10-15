
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


NUMERIC_COLS = ['delta','theta','alpha','beta','gamma','hr','hrv']


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()


    def fit_transform(self, df):
        X = df[NUMERIC_COLS].values
        Xs = self.scaler.fit_transform(X)
        return Xs, df['mood'].values, df['oracle_track'].values


    def transform(self, df):
        X = df[NUMERIC_COLS].values
        Xs = self.scaler.transform(X)
        return Xs

