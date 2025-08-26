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

try:
    # Download required NLTK data
    nlt.download('stopwords')
    stopwords = set(stopwords.words("english"))
    stemmer = nlt.stem.SnowballStemmer("english")

    def clean_data(text):
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stopwords]
        text = " ".join(text)
        text = [stemmer.stem(word) for word in text.split(' ')]
        text = " ".join(text)
        return text

    # Load dataset
    dataset = pd.read_csv(r"C:\HATE_SPEECH_DETECTION\twitter.csv")

    # Create labels
    dataset["labels"] = dataset["class"].map({0: "Hate Speech", 
                                            1: "Offensive Language", 
                                            2: "No Hate or Offensive Language"})

    # Create data DataFrame and clean tweets - Fixed warning
    data = dataset[["tweet", "labels"]].copy()
    data["tweet"] = data["tweet"].apply(clean_data)

    # Prepare features and labels
    X = np.array(data["tweet"])
    y = np.array(data["labels"])

    # Use TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = tfidf.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y)

    # Define parameter grid for DecisionTree
    param_grid = {
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    print("\n=== Training Model ===")
    # Initialize DecisionTree and GridSearch
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== Model Performance ===")
    print("\nBest Parameters:", grid_search.best_params_)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nAccuracy Score: {accuracy:.4f}")

    # Sample predictions
    print("\n=== Sample Predictions ===")
    sample_texts = [
        "The weather is really nice today",
        "This movie was amazing and inspiring",
        "Please help me understand this topic",
        "The food at this restaurant was terrible",
        "I disagree with your opinion but respect your view"
    ]

    print("\nPredictions:")
    print("-" * 80)
    print(f"{'Text':<50} | {'Predicted Category':<25}")
    print("-" * 80)

    for text in sample_texts:
        cleaned_sample = clean_data(text)
        sample_transformed = tfidf.transform([cleaned_sample])
        prediction = best_model.predict(sample_transformed)
        print(f"{text[:50]:<50} | {prediction[0]:<25}")

    print("-" * 80)

    # Visualize confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu")
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

except FileNotFoundError:
    print("Error: Could not find the dataset file. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {str(e)}")