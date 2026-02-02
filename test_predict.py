#!/usr/bin/env python3
"""
test_predict.py

Minimal unit tests for predict.py using pytest.

Run with:
  pytest test_predict.py -v
"""

import pytest
import json
import subprocess
from predict import load_model, predict_json


class TestLoadModel:
    def test_model_loads_successfully(self):
        pipe = load_model('models/tfidf_lr_pipeline.joblib')
        assert pipe is not None
        assert hasattr(pipe, 'predict_proba')
        assert hasattr(pipe, 'classes_')

    def test_model_has_3_classes(self):
        pipe = load_model('models/tfidf_lr_pipeline.joblib')
        assert len(pipe.classes_) == 3
        assert set(pipe.classes_) == {'Positive', 'Negative', 'Neutral'}


class TestPredictJSON:
    @pytest.fixture
    def model(self):
        return load_model('models/tfidf_lr_pipeline.joblib')

    def test_output_structure(self, model):
        out = predict_json(model, "I love this game!")
        assert isinstance(out, dict)
        assert 'sentiment' in out
        assert 'confidence' in out
        assert len(out) == 2

    def test_sentiment_values(self, model):
        out = predict_json(model, "I love this game!")
        assert out['sentiment'] in ['Positive', 'Negative', 'Neutral']

    def test_confidence_values(self, model):
        out = predict_json(model, "I love this game!")
        assert out['confidence'] in ['Low', 'Medium', 'High']

    def test_positive_example(self, model):
        out = predict_json(model, "I absolutely love this! Amazing!")
        assert out['sentiment'] == 'Positive'
        assert out['confidence'] in ['Medium', 'High']

    def test_negative_example(self, model):
        out = predict_json(model, "This is terrible and I hate it")
        assert out['sentiment'] == 'Negative'
        assert out['confidence'] in ['Medium', 'High']

    def test_neutral_example(self, model):
        out = predict_json(model, "Server maintenance scheduled for tonight")
        assert out['sentiment'] in ['Neutral', 'Negative']  # could be either depending on classifier

    def test_empty_string(self, model):
        out = predict_json(model, "")
        assert out['sentiment'] in ['Positive', 'Negative', 'Neutral']
        assert out['confidence'] in ['Low', 'Medium', 'High']

    def test_none_input(self, model):
        out = predict_json(model, None)
        assert out['sentiment'] in ['Positive', 'Negative', 'Neutral']
        assert out['confidence'] in ['Low', 'Medium', 'High']


class TestCLI:
    def test_cli_with_positional_arg(self):
        result = subprocess.run(
            ['python3', 'predict.py', 'I love this!'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        out = json.loads(result.stdout.strip())
        assert 'sentiment' in out
        assert 'confidence' in out

    def test_cli_with_stdin(self):
        result = subprocess.run(
            ['python3', 'predict.py'],
            input='This is awful\n',
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        out = json.loads(result.stdout.strip())
        assert 'sentiment' in out
        assert 'confidence' in out

    def test_cli_json_output_valid(self):
        result = subprocess.run(
            ['python3', 'predict.py', 'Test tweet'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        try:
            json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            pytest.fail("CLI output is not valid JSON")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
