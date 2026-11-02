import argparse

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME = "wine_quality"
PRODUCTION_ALIAS = "production"

mlflow.set_tracking_uri("http://localhost:5001")


def get_auc_for_version(client: MlflowClient, model_name: str, version: str) -> float:
    model_version = client.get_model_version(model_name, version)
    run = client.get_run(model_version.run_id)
    auc = run.data.metrics.get("auc")
    if auc is None:
        raise ValueError(
            f"Version {version} of '{model_name}' has no 'auc' metric logged "
            f"on its run ({model_version.run_id}); cannot evaluate for promotion."
        )
    return auc


def get_latest_version(client: MlflowClient, model_name: str) -> str:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions registered for model '{model_name}'.")
    latest = max(versions, key=lambda v: int(v.version))
    return latest.version


def promote_if_better(model_name: str, candidate_version: str) -> None:
    client = MlflowClient()

    candidate_auc = get_auc_for_version(client, model_name, candidate_version)
    print(f"Candidate: version {candidate_version}, AUC = {candidate_auc:.4f}")

    try:
        current_prod = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
        current_prod_version = current_prod.version
    except MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "not found" in str(e).lower():
            current_prod_version = None
        else:
            print(f"DEBUG: unexpected MlflowException while checking alias: {e}")
            raise

    if current_prod_version is None:
        print("No model currently holds the 'production' alias (bootstrap case).")
        client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, candidate_version)
        print(
            f"PROMOTED: version {candidate_version} is now '{PRODUCTION_ALIAS}' "
            f"(first production model)."
        )
        return

    if current_prod_version == candidate_version:
        print(
            f"Version {candidate_version} already holds the '{PRODUCTION_ALIAS}' "
            f"alias. Nothing to do."
        )
        return

    current_prod_auc = get_auc_for_version(client, model_name, current_prod_version)
    print(
        f"Current production: version {current_prod_version}, "
        f"AUC = {current_prod_auc:.4f}"
    )

    if candidate_auc > current_prod_auc:
        client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, candidate_version)
        print(
            f"PROMOTED: version {candidate_version} (AUC {candidate_auc:.4f}) "
            f"beats current production version {current_prod_version} "
            f"(AUC {current_prod_auc:.4f}). Alias moved."
        )
    else:
        print(
            f"REFUSED: version {candidate_version} (AUC {candidate_auc:.4f}) "
            f"does NOT beat current production version {current_prod_version} "
            f"(AUC {current_prod_auc:.4f}). Production alias left unchanged."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Model version to consider for promotion. Defaults to the latest registered version.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Registered model name (default: {MODEL_NAME})",
    )
    args = parser.parse_args()

    client = MlflowClient()
    version = args.version or get_latest_version(client, args.model_name)
    promote_if_better(args.model_name, version)


if __name__ == "__main__":
    main()