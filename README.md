# Twitter Sentiment Analysis

A machine learning system to classify Twitter sentiment into **Positive**, **Negative**, and **Neutral** categories using TF-IDF vectorization and Logistic Regression.

## Dataset

- **Training set:** 74,682 tweets (labeled)
- **Validation set:** 1,000 tweets (labeled)
- **Labels:** Positive, Negative, Neutral, Irrelevant (dropped in final model)
- **Sources:** Twitter game/software discussions across multiple topics (Borderlands, FIFA, CS-GO, etc.)

### Labeling Guidelines
- **Positive:** Praise, satisfaction, excitement, approval
- **Negative:** Frustration, sarcasm, anger, disappointment
- **Neutral:** Factual statements, announcements, mild complaints
- **Note:** Sarcasm overrides positive wording; do NOT infer intent beyond the text

## Model

**Architecture:** TF-IDF (1-2 grams) + Logistic Regression

**Performance on Validation Set:**
- **Accuracy:** 97.83%
- **Macro F1-score:** 0.9783
- **Precision/Recall per class:**
  - Negative: 98.48% / 97.74%
  - Neutral: 97.22% / 98.25%
  - Positive: 97.83% / 97.47%

**Training Details:**
- Feature extraction: TF-IDF with min_df=5, max_df=0.9, ngram_range=(1,2)
- Classifier: LogisticRegression (max_iter=1000, class_weight='balanced', solver='saga')
- Data preprocessing: lowercase, no special text cleaning

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/uday986675/twitter_sentiment_analysis2.git
cd twitter_sentiment_analysis2

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- joblib >= 1.0.0
- pytest >= 6.0.0 (for testing)

## Usage

### Command-line Prediction

Classify a single tweet:

```bash
# Positional argument
python predict.py "I love this game! So much fun :)"
# Output: {"sentiment": "Positive", "confidence": "High"}

# From stdin
echo "This update is terrible" | python predict.py
# Output: {"sentiment": "Negative", "confidence": "High"}
```

### Python API

```python
from predict import load_model, predict_json

# Load the model
pipe = load_model('models/tfidf_lr_pipeline.joblib')

# Predict sentiment
result = predict_json(pipe, "Your tweet text here")
print(result)
# Output: {"sentiment": "Positive", "confidence": "High"}
```

### JSON Output Format

All predictions return JSON:
```json
{
  "sentiment": "Positive | Negative | Neutral",
  "confidence": "Low | Medium | High"
}
```

**Confidence Levels:**
- **High:** probability >= 0.80
- **Medium:** probability >= 0.60 and < 0.80
- **Low:** probability < 0.60

## Testing

Run the unit test suite:

```bash
pytest test_predict.py -v
```

**Test coverage (13 tests):**
- Model loading and validation
- Prediction output structure and values
- Example classifications (positive, negative, neutral)
- Edge cases (empty strings, None input)
- CLI argument parsing and JSON output

## Project Structure

```
twitter_sentiment_analysis2/
├── predict.py                      # Inference script (CLI)
├── test_predict.py                 # Unit tests
├── app.py                          # Additional application file
├── models/
│   └── tfidf_lr_pipeline.joblib    # Trained model (TF-IDF + LogisticRegression)
├── twitter_training.csv            # Training dataset (74,682 tweets)
├── twitter_validation.csv          # Validation dataset (1,000 tweets)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Model Details

### Why TF-IDF + Logistic Regression?

1. **Interpretability:** Feature weights directly correspond to word importance
2. **Speed:** Fast training and inference
3. **Baseline:** Strong performance (97.83% accuracy) without heavy compute
4. **Production-ready:** Easy to serve, minimal dependencies

### Training Process

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=5, max_df=0.9)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga'))
])

pipe.fit(X_train, y_train)
```

### Key Hyperparameters
- **TfidfVectorizer:**
  - `ngram_range=(1,2)`: Unigrams + bigrams for context
  - `min_df=5`: Ignore terms appearing in < 5 documents
  - `max_df=0.9`: Ignore terms in > 90% of documents (likely stopwords)
- **LogisticRegression:**
  - `class_weight='balanced'`: Handle imbalanced class distribution
  - `solver='saga'`: Better for sparse data from TF-IDF

## Example Predictions

```bash
$ python predict.py "I absolutely love this! Amazing!!"
{"sentiment": "Positive", "confidence": "High"}

$ python predict.py "This is terrible and I hate it"
{"sentiment": "Negative", "confidence": "High"}

$ python predict.py "Server maintenance scheduled for tonight"
{"sentiment": "Neutral", "confidence": "Low"}

$ python predict.py "lol that was terrible but funny 😂"
{"sentiment": "Positive", "confidence": "Low"}
```

## Future Improvements

1. **Fine-tune Transformers:** DistilBERT or RoBERTa for higher accuracy (98%+)
2. **Text Preprocessing:** Handle emojis explicitly, clean URLs, normalize slang
3. **Ensemble Models:** Combine TF-IDF + Transformers for robustness
4. **Multi-label Classification:** Support mixed sentiments (e.g., "good game but bad servers")
5. **Aspect-based Sentiment:** Classify sentiment for specific game features
6. **Real-time API:** FastAPI/Flask wrapper for production deployment
7. **Model Versioning:** Track model iterations and performance metrics

## Model Persistence & Versioning

The trained model is saved using joblib:
```python
import joblib
pipe = joblib.load('models/tfidf_lr_pipeline.joblib')
```

**Note:** There are minor scikit-learn version mismatches (trained on 1.6.1, may load with 1.7.2+). This is safe but use the same version for production consistency.

## Performance Considerations

- **Inference latency:** ~1-5ms per tweet on CPU
- **Memory:** ~50MB (model file only)
- **No GPU required:** CPU-only compatible

## License

MIT License — Feel free to use for educational and commercial projects.

## Author

Uday  
GitHub: [@uday986675](https://github.com/uday986675)

## Contact & Support

For issues, questions, or improvements, open an issue on GitHub:  
[twitter_sentiment_analysis2/issues](https://github.com/uday986675/twitter_sentiment_analysis2/issues)

---

**Happy sentiment analyzing! 🎯**
