import numpy as np
import pandas as pd


def build_labels(df: pd.DataFrame) -> pd.Series:
    """Compute binary up/down labels for every row in df.

    Args:
        df: DataFrame with columns Close and Open. In the canonical pipeline
            this is the training slice of the aligned raw DataFrame
            (df.iloc[4:] after build_features has been called), so the index
            aligns directly with the output of build_features().

    Returns:
        Series of int (0 or 1) with the same length as df and a reset 0-based
        index. 1 means Close > Open for that row; 0 otherwise.
    """
    return (df["Close"] > df["Open"]).astype(int).reset_index(drop=True)


# `direction_labels` is an alias of `build_labels` kept for the regime/binary
# suites, which refer to the up/down target as the "direction" label.
def direction_labels(raw_align: pd.DataFrame) -> pd.Series:
    """Binary direction label: 1 if Close > Open, else 0 (== build_labels).

    Args:
        raw_align: Raw DataFrame aligned to the feature matrix (df.iloc[4:]).

    Returns:
        Int Series of {0, 1}, reset 0-based index.
    """
    return build_labels(raw_align)


def move_series(raw_align: pd.DataFrame) -> pd.Series:
    """Signed intrabar move (Close − Open) for each bar.

    Args:
        raw_align: Raw DataFrame aligned to the feature matrix (df.iloc[4:]).

    Returns:
        Float Series of Close − Open, reset 0-based index.
    """
    return (raw_align["Close"] - raw_align["Open"]).reset_index(drop=True)


def flat_mask(raw_align: pd.DataFrame) -> np.ndarray:
    """Boolean mask marking flat bars (Close == Open).

    Flat bars are an ambiguous up/down target and are dropped from the modelling
    set (train and test) via ``drop_flat``. Computed *after* features are built,
    so a flat bar still contributes to neighbouring bars' lags — only its own
    (ambiguous) target row is removed, never altering another row's features.

    Args:
        raw_align: Raw DataFrame aligned to the feature matrix (df.iloc[4:]).

    Returns:
        Boolean ndarray, True where Close == Open, same length as raw_align.
    """
    return (raw_align["Close"].values == raw_align["Open"].values)


def drop_flat(
    features: pd.DataFrame, raw_align: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop flat (Close == Open) rows from the aligned modelling set.

    Removes rows whose *target* bar is flat from both the feature matrix and the
    aligned raw frame (which carries the label/move/timestamp), before splitting.
    Features are built on the full series first, so a dropped flat bar still fed
    its neighbours' lags — no look-ahead, no feature change. Labels are therefore
    strictly binary 0/1 downstream.

    Args:
        features: Feature matrix aligned to ``raw_align`` (both 0-based, same length).
        raw_align: Raw DataFrame aligned to the feature matrix (df.iloc[4:]).

    Returns:
        (features, raw_align) with flat rows removed and indices reset 0-based.
    """
    keep = ~flat_mask(raw_align)
    n_before = len(features)
    features = features[keep].reset_index(drop=True)
    raw_align = raw_align[keep].reset_index(drop=True)
    print(f"drop_flat: removed {n_before - len(features):,} flat rows "
          f"({n_before:,} → {len(features):,})")
    return features, raw_align


if __name__ == "__main__":
    from pathlib import Path

    from src.features import build_features
    from src.load import load_raw
    from src.split import split

    df = load_raw(Path("data/raw/data.csv"))
    features = build_features(df)
    raw_align = df.iloc[4:].reset_index(drop=True)
    _, _ = split(features)
    raw_train, _ = split(raw_align)
    labels = build_labels(raw_train)
    print(f"Labels: {len(labels):,} rows")
    print(labels.value_counts().rename({1: "Up (1)", 0: "Down (0)"}))
