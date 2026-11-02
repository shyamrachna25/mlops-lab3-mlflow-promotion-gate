import time

import cloudpickle
import mlflow
import mlflow.pyfunc
import sklearn
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from data_prep import load_and_prepare_data
from train_baseline import SklearnModelWrapper, MODEL_NAME


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()

    with mlflow.start_run(run_name="deliberately_weak_model") as run:
        model = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=123)
        model.fit(X_train, y_train)

        predictions_test = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, predictions_test)
        mlflow.log_param("n_estimators", 1)
        mlflow.log_param("max_depth", 1)
        mlflow.log_metric("auc", auc_score)

        wrapped_model = SklearnModelWrapper(model)
        signature = infer_signature(X_train, wrapped_model.predict(None, X_train))
        conda_env = _mlflow_conda_env(
            additional_conda_deps=None,
            additional_pip_deps=[
                f"cloudpickle=={cloudpickle.__version__}",
                f"scikit-learn=={sklearn.__version__}",
            ],
            additional_conda_channels=None,
        )
        mlflow.pyfunc.log_model(
            "random_forest_model",
            python_model=wrapped_model,
            conda_env=conda_env,
            signature=signature,
        )
        run_id = run.info.run_id

    print(f"Deliberately weak model trained. AUC: {auc_score:.4f}")

    model_version = mlflow.register_model(
        f"runs:/{run_id}/random_forest_model", MODEL_NAME
    )
    time.sleep(5)
    print(f"Registered as '{MODEL_NAME}' version {model_version.version}")
    print(
        f"Now run: python promote_model.py --version {model_version.version}"
    )
    print("Expected: REFUSED, since this model's AUC is deliberately terrible.")


if __name__ == "__main__":
    main()