import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from main import download_link


def test_download_link():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    link = download_link(df, "test.csv")
    assert link is not None


def test_standard_scaler():
    scaler = StandardScaler()
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    scaled_data = scaler.fit_transform(data)
    assert scaled_data.shape == data.shape
