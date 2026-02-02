#!/usr/bin/env python3
"""
predict.py

Load the saved TF-IDF + Logistic Regression pipeline and predict sentiment for a single tweet.
Outputs JSON only, matching the required format:
{ "sentiment": "Positive|Negative|Neutral", "confidence": "Low|Medium|High" }

Usage:
  python predict.py "This game is awesome!"
  echo "This game is awful" | python predict.py
"""

import argparse
import json
import sys
import joblib
import pandas as pd

MODEL_DEFAULT = 'models/tfidf_lr_pipeline.joblib'


def load_model(path: str = MODEL_DEFAULT):
    return joblib.load(path)


def predict_json(pipe, text: str):
    text = text if text is not None else ''
    # Ensure string type
    if not isinstance(text, str):
        text = str(text)
    proba = pipe.predict_proba([text])[0]
    idx = int(proba.argmax())
    label = pipe.classes_[idx]
    score = float(proba[idx])

    if score >= 0.80:
        conf = 'High'
    elif score >= 0.60:
        conf = 'Medium'
    else:
        conf = 'Low'

    return {"sentiment": label, "confidence": conf}


def main():
    parser = argparse.ArgumentParser(description='Predict sentiment for a single tweet and print JSON.')
    parser.add_argument('text', nargs='?', help='Tweet text. If omitted, read from stdin.')
    parser.add_argument('-m', '--model', default=MODEL_DEFAULT, help='Path to saved pipeline (joblib file)')
    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        text = sys.stdin.read().strip()
        if not text:
            parser.error('No tweet text provided (positional arg or stdin).')

    try:
        pipe = load_model(args.model)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {e}"}))
        sys.exit(1)

    out = predict_json(pipe, text)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
