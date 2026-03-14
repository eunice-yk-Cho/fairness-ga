import io
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _download_text(url):
    with urlopen(url) as response:
        return response.read().decode("utf-8", errors="replace")


def download_compas():
    out_path = DATA_DIR / "compas.csv"
    if out_path.exists():
        print(f"[skip] exists: {out_path}")
        return

    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    raw = _download_text(url)
    df = pd.read_csv(io.StringIO(raw))

    cols = [
        "age",
        "sex",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_charge_degree",
        "two_year_recid",
    ]
    df = df[cols].dropna()
    df.to_csv(out_path, index=False)
    print(f"[ok] saved: {out_path}")


def download_german():
    out_path = DATA_DIR / "german_credit.csv"
    if out_path.exists():
        print(f"[skip] exists: {out_path}")
        return

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    raw = _download_text(url)

    columns = [
        "status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings",
        "employment",
        "installment_rate",
        "personal_status",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "other_installment",
        "housing",
        "existing_credits",
        "job",
        "people_liable",
        "telephone",
        "foreign_worker",
        "class",
    ]
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None, names=columns)

    sex_map = {
        "A91": "male",
        "A92": "female",
        "A93": "male",
        "A94": "male",
        "A95": "female",
    }
    df["sex"] = df["personal_status"].map(sex_map).fillna("unknown")
    # UCI Statlog german.data uses numeric labels (1=good, 2=bad).
    # Some variants may use A201/A202, so we support both.
    df["class"] = (
        df["class"]
        .astype(str)
        .map({"1": "good", "2": "bad", "A201": "good", "A202": "bad"})
    )
    df = df.drop(columns=["personal_status"]).dropna()
    df.to_csv(out_path, index=False)
    print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    download_compas()
    download_german()
