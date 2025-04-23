import pandas as pd
import os
import ast
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import seaborn as sns
from collections import Counter

nltk.download('punkt')

def load_csv(file_path):
    """Load a CSV file and return as DataFrame."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"File loaded successfully: {file_path}")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pd.errors.ParserError:
        print(f"Error: File '{file_path}' could not be parsed.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None

def remove_unnecessary_columns(df):
    """Remove specified columns from the DataFrame."""
    columns_to_remove = [
        "adult", "belongs_to_collection", "budget", "homepage", 
        "poster_path", "production_companies", "production_countries", 
        "revenue", "status", "video", "vote_average", "vote_count", "spoken_languages"
    ]
    df = df.drop(columns=columns_to_remove, errors='ignore')
    return df

def clean_keywords(keywords_str):
    """
    Convert keywords from JSON-like string to a clean comma-separated list.
    """
    if pd.isna(keywords_str) or keywords_str == '[]':
        return ""
    try:
        parsed = ast.literal_eval(keywords_str)
        return ", ".join(d['name'] for d in parsed if isinstance(d, dict) and 'name' in d)
    except Exception as e:
        print(f"Error parsing keywords: {keywords_str} - {e}")
        return ""

def merge_movies_with_keywords(movies_df, keywords_df):
    """Merge movie metadata with keywords on 'id'."""
    movies_df['id'] = movies_df['id'].astype(str)
    keywords_df['id'] = keywords_df['id'].astype(str)
    merged_df = pd.merge(movies_df, keywords_df, on='id')
    merged_df['keywords'] = merged_df['keywords'].apply(clean_keywords)
    return merged_df

def analyze_overview_length(df):
    """Analyze average sentence/token count in overview column."""
    if 'overview' not in df.columns:
        print("No 'overview' column found.")
        return

    sent_counts = []
    token_counts = []
    for text in df['overview'].dropna():
        sentences = sent_tokenize(text)
        tokens = word_tokenize(text)
        sent_counts.append(len(sentences))
        token_counts.append(len(tokens))

    print(f"Average Sentences per Overview: {np.mean(sent_counts):.2f}")
    print(f"Average Tokens per Overview: {np.mean(token_counts):.2f}")

def main():
    original_movies_file = "data/movies_metadata.csv"
    reduced_movies_file = "data/reduced_movies_metadata.csv"
    keywords_file = "data/keywords.csv"

    if not os.path.exists(reduced_movies_file):
        print(f"Reduced dataset not found. Creating {reduced_movies_file}...")
        original_df = load_csv(original_movies_file)
        if original_df is not None:
            reduced_df = remove_unnecessary_columns(original_df)
            reduced_df = reduced_df.head(10000)
            reduced_df.to_csv(reduced_movies_file, index=False)
            print(f"Reduced dataset saved to {reduced_movies_file}")
        else:
            print("Failed to create reduced dataset.")
            return

    movies_df = load_csv(reduced_movies_file)
    keywords_df = load_csv(keywords_file)

    if movies_df is not None and keywords_df is not None:
        merged_df = merge_movies_with_keywords(movies_df, keywords_df)
        print("Merged DataFrame:")
        print(merged_df.head())

        analyze_overview_length(merged_df)

        output_file = "data/enhanced_movies_dataset.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"Merged file saved to: {output_file}")

if __name__ == "__main__":
    main()
