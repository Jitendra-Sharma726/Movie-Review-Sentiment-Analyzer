import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

# --- Initial Setup: NLTK Stopwords ---
# A one-time check and download for the NLTK stopwords list.
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    # 1. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Split into words
    words = text.split()
    
    # 5. Remove stop words
    cleaned_words = [word for word in words if word not in STOP_WORDS]
    
    # 6. Join words back into a single string
    return ' '.join(cleaned_words)


def main():
    INPUT_FILE = 'IMDB Dataset.csv'
    OUTPUT_FILE = 'cleaned_imdb_dataset.csv'

    # 1. Load the dataset from INPUT_FILE.
    # Handle the case where the file might not be found.
    print(f"Loading data from '{INPUT_FILE}'...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Dataset not found at '{INPUT_FILE}'.")
        print("Please download it and place it in the correct directory.")
        return 

    # 2. Apply the cleaning function to the 'review' column.
    # Store the results in a new column named 'cleaned_review'.
    print("Applying text cleaning function to all reviews...")
    df['cleaned_review'] = df['review'].apply(clean_text)

    # 3. Prepare the final DataFrame for saving.
    # It should contain only the 'cleaned_review' and 'sentiment' columns.
    final_df = df[['cleaned_review', 'sentiment']]

    # 4. Save the final DataFrame to OUTPUT_FILE.
    # Ensure the CSV does not include the DataFrame index.
    print(f"Saving cleaned data to '{OUTPUT_FILE}'...")
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("Data cleaning process complete.")
    print(f"Cleaned data saved to {OUTPUT_FILE}")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()

