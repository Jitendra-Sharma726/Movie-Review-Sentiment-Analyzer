import pandas as pd
import re
from collections import Counter
from typing import Dict, List, Tuple, Any

def task_1_load_and_inspect(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        raise


def task_2_check_dataset_health(df: pd.DataFrame) -> Dict[str, int]:
    """
    Checks the total number of entries and missing values.
    """
    total_reviews = len(df)
    missing_reviews = df['review'].isnull().sum()
    missing_sentiments = df['sentiment'].isnull().sum()
    
    return {
        "total_reviews": int(total_reviews),
        "missing_reviews": int(missing_reviews),
        "missing_sentiments": int(missing_sentiments)
    }


def task_3_analyze_sentiment_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Counts the number of positive and negative reviews.
    """
    return df['sentiment'].value_counts()


def task_4_analyze_review_length(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates the average word count for all, positive, and negative reviews.
    """
    # FIX: Handle potential non-string (None/NaN) values in the 'review' column.
    # If x is not a string (e.g., it is None or NaN), its word count is considered 0.
    # This makes the function robust and prevents the "'NoneType' has no attribute 'split'" error.
    df['word_count'] = df['review'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    
    overall_average = df['word_count'].mean()
    average_positive = df[df['sentiment'] == 'positive']['word_count'].mean()
    average_negative = df[df['sentiment'] == 'negative']['word_count'].mean()
    
    return {
        "overall_average_word_count": overall_average,
        "average_positive_word_count": average_positive,
        "average_negative_word_count": average_negative
    }


# You can use the main block to run and test your functions
if __name__ == "__main__":
    try:
        # Define the path to the dataset
        FILE_PATH = 'IMDB Dataset.csv'

        # Task 1
        imdb_df = task_1_load_and_inspect(FILE_PATH)
        print("--- Task 1: Data Loaded ---")
        print(imdb_df.head())
        print("\n" + "="*50 + "\n")

        # Task 2
        print("--- Task 2: Checking Dataset Health ---")
        health_report = task_2_check_dataset_health(imdb_df)
        for key, value in health_report.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("\n" + "="*50 + "\n")

        # Task 3
        print("--- Task 3: Analyzing Sentiment Distribution ---")
        distribution = task_3_analyze_sentiment_distribution(imdb_df)
        print("Review counts per sentiment:")
        print(distribution)
        is_balanced = "Yes" if abs(distribution.get('positive', 0) - distribution.get('negative', 0)) < (0.05 * len(imdb_df)) else "No"
        print(f"\nIs the dataset balanced? {is_balanced}")
        print("\n" + "="*50 + "\n")

        # Task 4
        print("--- Task 4: Analyzing Review Length ---")
        length_analysis = task_4_analyze_review_length(imdb_df)
        for key, value in length_analysis.items():
            print(f"{key.replace('_', ' ').title()}: {value:.2f} words")
        print("\n" + "="*50 + "\n")

    except FileNotFoundError:
        print(f"Error: The file '{FILE_PATH}' was not found. Please make sure it's in the correct directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
