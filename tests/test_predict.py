import os
import pytest

from predict import load_model, predict_json

MODEL_PATH = 'models/tfidf_lr_pipeline.joblib'


def test_predict_json_structure():
    """Ensure predict_json returns the expected JSON keys and allowed values."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}")

    pipe = load_model(MODEL_PATH)
    out = predict_json(pipe, "I love this game! So much fun :)")

    assert isinstance(out, dict)
    assert 'sentiment' in out and 'confidence' in out
    assert out['sentiment'] in {'Positive', 'Negative', 'Neutral'}
    assert out['confidence'] in {'Low', 'Medium', 'High'}
