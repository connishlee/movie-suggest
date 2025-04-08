import pandas as pd

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print("File loaded successfully!")
        print(df.head())  # Shows the first 5 rows
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pd.errors.ParserError:
        print(f"Error: File '{file_path}' could not be parsed.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    file_path = "datasets/the-movies-dataset/movies_metadata.csv"
    load_csv(file_path)