import pandas as pd
import numpy as np
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
import nltk as nlt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from typing import List, Tuple, Dict, Any
import logging
import joblib
import os

class TextPreprocessor:
    """Class for text preprocessing operations"""
    
    def __init__(self):
        self._download_nltk_data()
        self.stopwords = set(stopwords.words("english"))
        self.stemmer = nlt.stem.SnowballStemmer("english")
    
    def _download_nltk_data(self) -> None:
        """Download required NLTK data"""
        nlt.download('stopwords')
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data"""
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in self.stopwords]
        text = " ".join(text)
        text = [self.stemmer.stem(word) for word in text.split(' ')]
        return " ".join(text)

class HateSpeechClassifier:
    """Main class for hate speech classification"""
    
    def __init__(self, model_path: str = "models"):
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.tfidf = None
        self.model_path = model_path
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hate_speech_classifier.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and prepare the dataset"""
        try:
            dataset = pd.read_csv(filepath)
            dataset["labels"] = dataset["class"].map({
                0: "Hate Speech",
                1: "Offensive Language",
                2: "No Hate or Offensive Language"
            })
            return dataset
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        data_clean = data[["tweet", "labels"]].copy()
        data_clean["tweet"] = data_clean["tweet"].apply(self.preprocessor.clean_text)
        X = np.array(data_clean["tweet"])
        y = np.array(data_clean["labels"])
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        try:
            logging.info("Starting model training...")
            
            # TF-IDF Vectorization
            self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
            X = self.tfidf.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model with GridSearch
            param_grid = {
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            dt = DecisionTreeClassifier(random_state=42)
            grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            y_pred = self.model.predict(X_test)
            
            # Log results
            logging.info(f"Best Parameters: {grid_search.best_params_}")
            logging.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
            logging.info(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
            
            # Save confusion matrix
            self.plot_confusion_matrix(y_test, y_pred)
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, texts: List[str]) -> List[str]:
        """Predict classifications for new texts"""
        try:
            cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
            vectors = self.tfidf.transform(cleaned_texts)
            predictions = self.model.predict(vectors)
            return predictions
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10,8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu")
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def save_model(self) -> None:
        """Save the trained model and vectorizer"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        joblib.dump(self.model, f"{self.model_path}/model.joblib")
        joblib.dump(self.tfidf, f"{self.model_path}/tfidf.joblib")
    
    def load_model(self) -> None:
        """Load a trained model and vectorizer"""
        self.model = joblib.load(f"{self.model_path}/model.joblib")
        self.tfidf = joblib.load(f"{self.model_path}/tfidf.joblib")
