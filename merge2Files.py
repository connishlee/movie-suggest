# this script at least merges the two and cuts down to 10l

import pandas as pd
import os

def load_csv(file_path):
    """Load a CSV file and return as DataFrame."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"File loaded successfully: {file_path}")
        print(f"Shape: {df.shape}")
        print(df.head(3))  # Shows the first 3 rows
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pd.errors.ParserError:
        print(f"Error: File '{file_path}' could not be parsed.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None

def merge_movies_with_keywords(movies_file, keywords_file, output_file):
    """
    Merge the reduced movies metadata with keywords file based on matching IDs.
    
    Parameters:
    -----------
    movies_file : str
        Path to the reduced movies metadata CSV file
    keywords_file : str
        Path to the keywords CSV file
    output_file : str
        Path where the merged CSV will be saved
    
    Returns:
    --------
    pandas.DataFrame
        The merged dataframe
    """
    # Load the CSV files
    movies_df = load_csv(movies_file)
    keywords_df = load_csv(keywords_file)
    
    if movies_df is None or keywords_df is None:
        print("Could not load one or both input files.")
        return None
    
    # Ensure ID columns have the same data type before merging
    # Convert ID columns to strings to ensure proper matching
    if 'id' in movies_df.columns:
        movies_df['id'] = movies_df['id'].astype(str)
    
    if 'id' in keywords_df.columns:
        keywords_df['id'] = keywords_df['id'].astype(str)
    
    # Merge the dataframes on the 'id' column
    print("Merging movies with keywords...")
    merged_df = pd.merge(
        movies_df, 
        keywords_df, 
        on='id',
        how='left'  # Keep all movies, even those without keywords
    )
    
    print(f"Merged shape: {merged_df.shape}")
    
    # Check for movies without keywords
    missing_keywords = merged_df[merged_df['keywords'].isna()].shape[0]
    if missing_keywords > 0:
        print(f"Note: {missing_keywords} movies ({missing_keywords/len(merged_df)*100:.1f}%) don't have matching keywords.")
    
    # Save the merged dataframe
    merged_df.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file}")
    
    return merged_df

if __name__ == "__main__":
    # Paths for input and output files
    data_dir = "data"
    movies_file = os.path.join(data_dir, "movies_metadata_10k.csv")
    keywords_file = os.path.join(data_dir, "keywords.csv")
    output_file = os.path.join(data_dir, "mov_with_keywords_10k.csv")
    
    # Merge movies with keywords
    merged_df = merge_movies_with_keywords(movies_file, keywords_file, output_file)