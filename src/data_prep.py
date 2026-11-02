import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 123


def load_and_prepare_data(data_dir: str = "data"):
    white_wine = pd.read_csv(f"{data_dir}/winequality-white.csv", sep=";")
    red_wine = pd.read_csv(f"{data_dir}/winequality-red.csv", sep=",")

    red_wine["is_red"] = 1
    white_wine["is_red"] = 0

    data = pd.concat([red_wine, white_wine], axis=0)
    data.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)

    # Define "high quality" as quality >= 7 (binary classification target)
    data.quality = (data.quality >= 7).astype(int)

    X = data.drop(["quality"], axis=1)
    y = data.quality

    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, train_size=0.6, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=0.5, random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test