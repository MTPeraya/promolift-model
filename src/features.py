"""Backwards compatibility shim for features."""

from promolift.features import DEFAULT_SEGMENT_MAPPING, compute_rfm_features

__all__ = ["DEFAULT_SEGMENT_MAPPING", "compute_rfm_features"]

if __name__ == "__main__":
    from data_loader import load_data
    try:
        data = load_data()
        features = compute_rfm_features(data["transactions"], data["customers"])
        print("Features computed successfully!")
        print(features.head())
        print(features.describe())
    except Exception as e:  # noqa: BLE001
        print(f"Error computing features: {e}")
