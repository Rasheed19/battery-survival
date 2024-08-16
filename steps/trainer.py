import pandas as pd
import numpy as np
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from utils.generic_helper import score_survival_model
from utils.data_wrangler import DataFrameCaster


def model_trainer(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    parameter_space: dict,
    cv: None | int | KFold = None,
    fit_params: dict | None = None,
) -> tuple[Pipeline, float]:

    if cv is None:
        cv = KFold(n_splits=5, random_state=42, shuffle=True)

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", RobustScaler()),
            ("cast", DataFrameCaster(X_train.columns)),
            ("sksurv", GradientBoostingSurvivalAnalysis()),
        ]
    )

    gcv = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_space,
        scoring=score_survival_model,
        cv=cv,
        refit=True,
    )

    if fit_params is None:
        gcv.fit(X_train, y_train)
    else:
        gcv.fit(X_train, y_train, **fit_params)

    return gcv.best_estimator_, gcv.best_score_
