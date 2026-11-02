import time

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score

from data_prep import load_and_prepare_data

MODEL_NAME = "wine_quality"
MAX_EVALS = 20  # reduced from the original's 96 since this runs sequentially
SEARCH_SEED = 123  # same seed used everywhere else in this project

search_space = {
    "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
    "learning_rate": hp.loguniform("learning_rate", -3, 0),
    "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
    "objective": "binary:logistic",
    "seed": 123,
}


def make_train_model(X_train, y_train, X_val, y_val):
    def train_model(params):
        mlflow.xgboost.autolog()
        with mlflow.start_run(nested=True):
            train = xgb.DMatrix(data=X_train, label=y_train)
            validation = xgb.DMatrix(data=X_val, label=y_val)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(validation, "validation")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            validation_predictions = booster.predict(validation)
            auc_score = roc_auc_score(y_val, validation_predictions)
            mlflow.log_metric("auc", auc_score)

            signature = infer_signature(X_train, booster.predict(train))
            mlflow.xgboost.log_model(booster, "model", signature=signature)

            return {"status": STATUS_OK, "loss": -1 * auc_score}

    return train_model


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    train_model = make_train_model(X_train, y_train, X_val, y_val)
    trials = Trials()

    with mlflow.start_run(run_name="xgboost_models") as parent_run:
        fmin(
            fn=train_model,
            space=search_space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=trials,
            rstate=np.random.default_rng(SEARCH_SEED),
        )
        parent_run_id = parent_run.info.run_id
        
    child_runs = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=["metrics.auc DESC"],
    )
    best_run = child_runs.iloc[0]
    best_auc = best_run["metrics.auc"]
    print(f"Best challenger trial AUC: {best_auc:.4f}")
    print(f"Best run ID: {best_run.run_id}")

    model_version = mlflow.register_model(
        f"runs:/{best_run.run_id}/model", MODEL_NAME
    )
    time.sleep(5)

    print(f"Registered as '{MODEL_NAME}' version {model_version.version}")
    print("NOTE: this model is registered but NOT yet promoted.")
    print("Run promote_model.py to decide whether it should become production.")


if __name__ == "__main__":
    main()