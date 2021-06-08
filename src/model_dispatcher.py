from sklearn import tree
from sklearn import ensemble
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

models = {
    "rf": ensemble.RandomForestRegressor(n_jobs=-1, n_estimators=1200, max_depth=None, max_features=None,
                min_samples_split=2, random_state=42, verbose=1),
    'xgb': XGBRegressor(random_state = 42, n_jobs = -1, verbose=1),
    'lgbm': LGBMRegressor(random_state = 42, n_jobs = -1, verbose=1)
}