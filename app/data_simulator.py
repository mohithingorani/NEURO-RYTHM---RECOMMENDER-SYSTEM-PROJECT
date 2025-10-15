import os
focus = max(0, 1 - stress - fatigue + np.random.normal(0, 0.05))

# create features influenced by states
band = base + np.array([0.1*fatigue, 0.05*stress, 0.2*focus, 0.1*stress, 0.05*focus])
band = band / band.sum()

hr = int(60 + 40*stress + np.random.normal(0, 3))
hrv = max(10, 80 - 30*stress + np.random.normal(0, 5))

# label mood as one of: relaxed, anxious, focused, fatigued
if stress > 0.6:
    mood = 'anxious'
elif focus > 0.6:
    mood = 'focused'
elif fatigue > 0.6:
    mood = 'fatigued'
else:
    mood = 'relaxed'

# choose a "therapeutic" track (oracle) for supervised training
if mood == 'anxious':
    target = 't4'
elif mood == 'focused':
    target = 't2'
elif mood == 'fatigued':
    target = 't1'
else:
    target = 't3'

rows.append({
    'session_id': f's{i}',
    'delta': band[0],
    'theta': band[1],
    'alpha': band[2],
    'beta': band[3],
    'gamma': band[4],
    'hr': hr,
    'hrv': hrv,
    'mood': mood,
    'oracle_track': target
})

DF = pd.DataFrame(rows)
DF.to_csv(DATA_DIR / 'simulated_sessions.csv', index=False)
print('Wrote', DATA_DIR / 'simulated_sessions.csv')
