import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
np.random.seed(0)

### BEST MODEL: 
# XGBClassifier(booster='gbtree', n_estimators=200, learning_rate=0.01, max_depth=30, use_label_encoder=False)

# SETUP
train = pd.read_csv('jane_st.csv')
train = train.dropna()
X = train[[str(c) for c in train.columns if 'feature' in c] + ['weight'] + ['date'] + ['ts_id']]
y = pd.Series([1 if x > 0 else 0 for x in train['resp']])

# GRID SEARCH (determined max_depth separetly to save time)
model = XGBClassifier(use_label_encoder=False)
n_estimators = [100, 200, 300, 400, 500]
learning_rate = [0.001, 0.01, 0.1, 0.2]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=False)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, y)

# RESULTS
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

# PLOT
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
# plot results
scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
for i, value in enumerate(learning_rate):
    plt.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Log Loss')
plt.title('Sample = 85k')
plt.savefig('n_estimators_vs_learning_rate.png')
