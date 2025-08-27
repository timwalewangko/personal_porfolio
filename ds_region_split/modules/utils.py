import pandas as pd


def create_features(df, label=None):
    """
    Creates features from index
    """
    df["date"] = df.index
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekofyear"] = df["date"].dt.isocalendar().week

    X = df[
        [
            "quarter",
            "month",
            "year",
            "weekofyear",
        ]
    ]
    if label:
        y = df[label]
        return X, y
    return X
