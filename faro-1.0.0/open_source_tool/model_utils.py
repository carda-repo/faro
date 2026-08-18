import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_model(model_path: str | Path = "faro.joblib") -> Any:
    print(
        "Loading model... Depending on your system, this may take a few seconds.",
        flush=True,
    )
    return joblib.load(model_path)


def build_feature_frame(model: Any) -> pd.DataFrame:
    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        raise ValueError("The model does not expose feature_names_in_.")
    return pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)


def load_metadata(metadata_path: str | Path = "faro.json") -> dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def classify_score(score: float, metadata: dict[str, Any]) -> str:
    prediction_metadata = metadata["prediction"]
    broad_screening_threshold = float(
        prediction_metadata["broad_screening_threshold"]["value"]
    )
    f1_optimal_threshold = float(
        prediction_metadata["f1_optimal_threshold"]["value"]
    )

    if score >= f1_optimal_threshold:
        return "high_priority_risk_signal"
    if score >= broad_screening_threshold:
        return "risk_signal"
    return "no_signal"


def predict_with_model(
    model: Any,
    features: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(features) != 1:
        raise ValueError("predict_with_model() expects exactly one feature row.")

    if metadata is None:
        metadata = load_metadata()

    prediction_metadata = metadata["prediction"]
    broad_screening_threshold = float(
        prediction_metadata["broad_screening_threshold"]["value"]
    )
    f1_optimal_threshold = float(
        prediction_metadata["f1_optimal_threshold"]["value"]
    )
    score = float(model.predict_proba(features)[0, 1])

    return {
        "prediction": int(score >= broad_screening_threshold),
        "score": score,
        "risk_band": classify_score(score, metadata),
        "broad_screening_threshold": broad_screening_threshold,
        "f1_optimal_threshold": f1_optimal_threshold,
    }
