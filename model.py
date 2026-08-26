import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import re

# --- File Paths ---
INPUT_FILE = 'cleaned_imdb_dataset.csv'
OUTPUT_FILE = 'sentiment_model.pkl'

def train_model():
    """Main function to train and save the sentiment analysis model."""
    print("--- Starting Model Training ---")
    try:
        # --- Task 1: Load the Data ---
        df = pd.read_csv(INPUT_FILE)

        # --- Task 2: Prepare Features and Target ---
        X = df['cleaned_review']
        y = df['sentiment']

        # --- Task 3: Create Training and Testing Sets ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Task 4: Vectorize the Text ---
        vectorizer = CountVectorizer()
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # --- Task 5: Train the Model ---
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        # --- Task 6: Evaluate Performance ---
        predictions = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        # --- Task 7: Save the Artifacts ---
        artifacts = {'vectorizer': vectorizer, 'model': model}
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(artifacts, f)

        print(f"--- Model and Vectorizer saved successfully to {OUTPUT_FILE}! ---")
    except FileNotFoundError:
        print(f"Error: The input file '{INPUT_FILE}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Prediction Example (This part is complete for you) ---

def clean_text(text):
    """A simple function to clean text for prediction."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
    return text

def run_prediction_example():
    """Loads the saved model and runs predictions on sample reviews."""
    print("\n--- Running Prediction Example ---")
    try:
        with open(OUTPUT_FILE, 'rb') as f:
            artifacts = pickle.load(f)
        
        vectorizer = artifacts['vectorizer']
        model = artifacts['model']
        
        sample_reviews = [
            "This movie was absolutely fantastic, one of the best I have seen all year!",
            "A complete waste of time. The plot was boring and the acting was terrible.",
            "It was an okay film, not great but not bad either."
        ]
        
        for review in sample_reviews:
            # New, unseen data must go through the same steps: clean and transform
            cleaned_review = clean_text(review)
            review_vec = vectorizer.transform([cleaned_review])
            prediction = model.predict(review_vec)[0] # [0] to get the label from the array
            
            print(f"\nReview: '{review}'")
            print(f"Predicted Sentiment: {prediction}")
            
    except FileNotFoundError:
        print(f"Could not load '{OUTPUT_FILE}'. Please train the model first by running the script.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    train_model()
    run_prediction_example()
