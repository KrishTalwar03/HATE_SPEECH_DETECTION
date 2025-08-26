# Hate Speech Detection

A machine learning project for detecting hate speech in text using Decision Trees.

## Features

- Text preprocessing with NLTK
- TF-IDF vectorization
- Decision Tree classifier with GridSearchCV
- Confusion matrix visualization
- Model persistence
- Logging system
- Object-oriented design

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_classifier.py
```

## Project Structure

- `hate_speech_classifier.py`: Main classifier implementation
- `run_classifier.py`: Script to run the classifier
- `models/`: Saved model files
- `confusion_matrix.png`: Visualization output
- `hate_speech_classifier.log`: Log file

## Requirements

Create a requirements.txt file:

```
pandas
numpy
scikit-learn
nltk
seaborn
matplotlib
joblib
```

## License

MIT

## Author

Krish Talwar