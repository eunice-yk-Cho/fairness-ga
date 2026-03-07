import os
import pandas as pd

# data 디렉터리는 프로젝트 루트 기준 (스크립트 위치와 무관)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(_DATA_DIR, "adult.csv")

def download_adult_dataset():

    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]

    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    print("Downloading dataset from UCI...")

    train = pd.read_csv(
        train_url,
        header=None,
        names=columns,
        skipinitialspace=True
    )

    test = pd.read_csv(
        test_url,
        header=0,
        names=columns,
        skipinitialspace=True
    )

    df = pd.concat([train, test], ignore_index=True)

    # Remove trailing "." in income column (from test set)
    df["income"] = df["income"].str.replace(".", "", regex=False)

    os.makedirs(_DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    download_adult_dataset()