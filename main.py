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

def remove_columns(input_file, output_file, columns_to_remove):
    """
    Remove specified columns from a CSV file and save the result.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str
        Path where the output CSV will be saved
    columns_to_remove : list
        List of column names to remove
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with specified columns removed
    """
    df = load_csv(input_file)
    
    if df is None:
        print("Could not load the input file.")
        return None
    
    print(f"Original columns ({len(df.columns)}):")
    print(df.columns.tolist())
    
    columns_to_remove_existing = [col for col in columns_to_remove if col in df.columns]
    columns_not_found = [col for col in columns_to_remove if col not in df.columns]
    
    if columns_not_found:
        print(f"Warning: These columns were not found in the input file: {columns_not_found}")
    
    if not columns_to_remove_existing:
        print("No columns to remove were found in the input file.")
        return df
    
    df_reduced = df.drop(columns=columns_to_remove_existing)
    
    print(f"Removed {len(columns_to_remove_existing)} columns: {columns_to_remove_existing}")
    print(f"Remaining columns ({len(df_reduced.columns)}):")
    print(df_reduced.columns.tolist())
    
    df_reduced.to_csv(output_file, index=False)
    print(f"File with removed columns saved to {output_file}")
    
    return df_reduced

def process_movies_dataset(input_file, keywords_file, final_output_file, columns_to_remove):
    """
    Complete processing pipeline:
    1. Remove specified columns
    2. Merge with keywords file
    
    Parameters:
    -----------
    input_file : str
        Path to the input movies CSV file
    keywords_file : str
        Path to the keywords CSV file
    final_output_file : str
        Path where final processed CSV will be saved
    columns_to_remove : list
        List of column names to remove
    """
    temp_file = os.path.join(os.path.dirname(final_output_file), "temp_reduced_columns.csv")
    
    print("\n=== STEP 1: REMOVING UNWANTED COLUMNS ===")
    reduced_df = remove_columns(input_file, temp_file, columns_to_remove)
    
    if reduced_df is None:
        print("Column removal failed. Stopping process.")
        return None
    
    # Step 2: Merge with keywords
    print("\n=== STEP 2: MERGING WITH KEYWORDS ===")
    final_df = merge_movies_with_keywords(temp_file, keywords_file, final_output_file)
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Temporary file {temp_file} removed.")
    
    print("\n=== PROCESSING COMPLETE ===")
    print(f"Final dataset saved to {final_output_file}")
    
    if final_df is not None:
        print(f"Final columns ({len(final_df.columns)}):")
        print(final_df.columns.tolist())
    
    return final_df

if __name__ == "__main__":
    data_dir = "data"
    input_file = os.path.join(data_dir, "movies_metadata_10k.csv")
    keywords_file = os.path.join(data_dir, "keywords.csv")
    final_output_file = os.path.join(data_dir, "updated_movies_metadata.csv")
    
    # Columns to remove
    columns_to_remove = [
        "adult", "belongs_to_collection", "budget", "homepage", 
        "poster_path", "production_companies", "production_countries", 
        "revenue", "status", "video", "vote_average", "vote_count", "spoken_languages"
    ]
    
    # Run the complete processing pipeline
    processed_df = process_movies_dataset(
        input_file, 
        keywords_file, 
        final_output_file, 
        columns_to_remove
    )