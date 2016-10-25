import util
import pandas as pd


def test_randomize_data():
    df = pd.DataFrame([[0],[1], [2], [3]])
    original_index = df.index.values

    rand_df = util.randomize_data(df)
    rand_index = rand_df.index.values

    assert rand_index[0] != original_index[0]
    assert rand_index[1] != original_index[1]
    assert rand_index[2] != original_index[2]
    assert rand_index[3] != original_index[3]


def test_standardize_data():
    df = pd.DataFrame([[2, 4, 1], [4, 3, 1], [0, 2, 2]], columns=["F1", "F2", "Label"])

    expected_mean = pd.Series({"F1": 2.0, "F2": 3.0})
    expected_std = pd.Series({"F1": 2.0, "F2": 1.0})
    std_df, actual_mean, actual_std = util.standardize_data(df[df.columns[0:-1]])

    assert expected_mean["F1"] == actual_mean["F1"]
    assert expected_mean["F2"] == actual_mean["F2"]
    assert expected_std["F1"] == actual_std["F1"]
    assert expected_std["F2"] == actual_std["F2"]

    assert std_df.iloc[0]["F1"] == (2 - 2.0) / 2.0
    assert std_df.iloc[0]["F2"] == (4 - 3.0) / 1.0

    assert std_df.iloc[1]["F1"] == (4 - 2.0) / 2.0
    assert std_df.iloc[1]["F2"] == (3 - 3.0) / 1.0

    assert std_df.iloc[2]["F1"] == (0 - 2.0) / 2.0
    assert std_df.iloc[2]["F2"] == (2 - 3.0) / 1.0


def test_split_data():
    pass