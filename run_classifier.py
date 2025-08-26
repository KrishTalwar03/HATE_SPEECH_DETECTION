from hate_speech_classifier import HateSpeechClassifier

def main():
    # Initialize classifier
    classifier = HateSpeechClassifier()
    
    # Load and prepare data
    data = classifier.load_data("twitter.csv")
    X, y = classifier.prepare_data(data)
    
    # Train model
    classifier.train(X, y)
    
    # Save model
    classifier.save_model()
    
    # Test with sample texts
    sample_texts = [
        "The weather is really nice today",
        "This movie was amazing and inspiring",
        "Please help me understand this topic",
        "The food at this restaurant was terrible",
        "I disagree with your opinion but respect your view"
    ]
    
    predictions = classifier.predict(sample_texts)
    
    print("\n=== Sample Predictions ===")
    print("-" * 80)
    print(f"{'Text':<50} | {'Predicted Category':<25}")
    print("-" * 80)
    
    for text, pred in zip(sample_texts, predictions):
        print(f"{text[:50]:<50} | {pred:<25}")
    print("-" * 80)

if __name__ == "__main__":
    main()