import requests

from data_prep import load_and_prepare_data

URL = "http://localhost:5002/invocations"


def main():
    _, _, X_test, _, _, y_test = load_and_prepare_data()

    # Send a small sample, not the whole test set, to keep the request light
    sample = X_test.head(5)
    payload = {"dataframe_split": sample.to_dict(orient="split")}

    response = requests.post(URL, json=payload)
    response.raise_for_status()
    predictions = response.json()

    print("Sample true labels:", y_test.head(5).tolist())
    print("Model predictions:  ", predictions)


if __name__ == "__main__":
    main()