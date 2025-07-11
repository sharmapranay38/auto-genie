from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Removed for speed
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.svm import SVC, SVR  # Removed for speed
from xgboost import XGBClassifier, XGBRegressor
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    has_lightgbm = True
except ImportError:
    has_lightgbm = False
from tqdm import tqdm

def train_and_return_models(X_train, y_train, task_type):
    models = {}
    model_defs = []
    if task_type == 'classification':
        model_defs = [
            ('LogisticRegression', LogisticRegression(max_iter=1000)),
            # ('RandomForest', RandomForestClassifier()),
            ('XGBoost', XGBClassifier(eval_metric='logloss')),
            ('KNN', KNeighborsClassifier()),
            # ('SVM', SVC(probability=True)),
        ]
        if has_lightgbm:
            model_defs.append(('LightGBM', LGBMClassifier()))
    else:
        model_defs = [
            ('LinearRegression', LinearRegression()),
            # ('RandomForest', RandomForestRegressor()),
            ('XGBoost', XGBRegressor()),
            ('KNN', KNeighborsRegressor()),
            # ('SVM', SVR()),
        ]
        if has_lightgbm:
            model_defs.append(('LightGBM', LGBMRegressor()))
    for name, model in tqdm(model_defs, desc='Training models'):
        models[name] = model.fit(X_train, y_train)
    return models 