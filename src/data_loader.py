"""Backwards compatibility shim for data_loader."""

from promolift.data_loader import load_raw_data


def load_data(data_dir="data"):
    """
    Loads all datasets from the designated data directory.
    Returns:
        dict: A dictionary of Pandas DataFrames.
    """
    return load_raw_data(data_dir=data_dir, validate=False)


if __name__ == "__main__":
    try:
        data = load_data()
        for k, v in data.items():
            print(f"Loaded {k}: {v.shape}")
    except Exception as e:  # noqa: BLE001
        print(f"Error loading data: {e}")
