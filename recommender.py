import pandas as pd
import numpy as np
import ast
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class MovieRecommender:
    def __init__(self, num_clusters=10):
        """Initialize the movie recommender with necessary components"""
        print("Initializing MovieRecommender...")
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.cluster_classifier = None  # Will be a classifier to predict clusters
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.num_clusters = num_clusters
        self.nn_model = None  # For finding similar movies
        self.movie_embeddings = None
        self.movies_df = None
        self.user_preferences = {}
        
    def load_data(self, file_path):
        """Load movie data from CSV file"""
        print(f"Loading data from {file_path}...")
        self.movies_df = pd.read_csv(file_path)
        
        # Convert genres from string representation to list (for reference only)
        self.movies_df['genres'] = self.movies_df['genres'].apply(self._parse_genres)
        
        # Clean up overview column
        self.movies_df['overview'] = self.movies_df['overview'].fillna('')
        
        # Remove rows with empty overviews
        self.movies_df = self.movies_df[self.movies_df['overview'].str.strip() != '']
        
        print(f"Loaded {len(self.movies_df)} movies from dataset")
        return self.movies_df
    
    def _parse_genres(self, genres_str):
        """Parse genres from JSON-like string to list of genre names"""
        if pd.isna(genres_str) or genres_str == '[]':
            return []
        
        try:
            # Parse the string into a Python object
            genres_list = ast.literal_eval(genres_str)
            # Extract just the genre names
            return [genre['name'] for genre in genres_list if isinstance(genre, dict) and 'name' in genre]
        except Exception as e:
            # print(f"Error parsing genres: {genres_str} - {e}")
            return []
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if isinstance(text, str) and text.strip():
            # Convert to lowercase
            text = text.lower()
            # Remove special characters and numbers
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            words = text.split()
            filtered_words = [word for word in words if word not in stop_words]
            return ' '.join(filtered_words)
        return ''
    
    def extract_features(self, texts):
        """Extract features using transformer model"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Tokenize and prepare for model
        encoded_inputs = self.tokenizer(processed_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Use [CLS] token embedding
            
        return embeddings
    
    def create_clusters(self):
        """Create clusters based on overviews using K-means"""
        print(f"Extracting features from movie overviews for clustering...")
        self.movie_embeddings = self.extract_features(self.movies_df['overview'].tolist())
        
        print(f"Clustering movie overviews into {self.num_clusters} categories...")
        cluster_labels = self.kmeans.fit_predict(self.movie_embeddings)
        
        # Add cluster labels to dataframe
        self.movies_df['cluster'] = cluster_labels
        
        # Count movies in each cluster
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        print("Movies per cluster:")
        for cluster_id, count in cluster_counts.items():
            print(f"Cluster {cluster_id}: {count} movies")
            
        # Create a nearest neighbors model for finding similar movies
        print("Creating nearest neighbors model...")
        self.nn_model = NearestNeighbors(n_neighbors=20, algorithm='auto', metric='cosine')
        self.nn_model.fit(self.movie_embeddings)
        
        return cluster_labels
    
    def train_cluster_classifier(self):
        """Train a classifier to predict clusters based on movie overviews"""
        if 'cluster' not in self.movies_df.columns:
            print("Clusters not found. Creating clusters first...")
            self.create_clusters()
        
        print("Training movie category classifier...")
        # Get feature embeddings from descriptions
        features = self.movie_embeddings
        targets = self.movies_df['cluster'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Train the classifier
        self.cluster_classifier = LogisticRegression(
            max_iter=1000, 
            multi_class='multinomial',
            solver='lbfgs'
        )
        self.cluster_classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.cluster_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Cluster classification metrics:")
        print(f"- Accuracy: {accuracy:.4f}")
        print(f"- F1 Score: {f1:.4f}")
        
        return accuracy
    
    def classify_text(self, text):
        """Classify a text into one of the movie clusters"""
        # Extract features
        features = self.extract_features([text])
        
        # Predict cluster
        cluster = self.cluster_classifier.predict(features)[0]
        
        # Get prediction probabilities for all clusters
        probs = self.cluster_classifier.predict_proba(features)[0]
        
        return cluster, probs
    
    def get_cluster_keywords(self, cluster_id, top_n=20):
        """Get most common words in a cluster to characterize it"""
        # Get all movie overviews from this cluster
        cluster_overviews = self.movies_df[self.movies_df['cluster'] == cluster_id]['overview']
        
        # Combine and tokenize them
        all_words = ' '.join(cluster_overviews).lower()
        all_words = re.sub(r'[^\w\s]', '', all_words)
        words = all_words.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Return top N words
        return word_counts.most_common(top_n)
    
    def find_similar_movies(self, text=None, movie_idx=None, n=10):
        """Find movies with similar overviews"""
        if text is not None:
            # Get embedding for the text
            query_vector = self.extract_features([text])[0].reshape(1, -1)
        elif movie_idx is not None:
            # Use embedding for the specified movie
            query_vector = self.movie_embeddings[movie_idx].reshape(1, -1)
        else:
            raise ValueError("Either text or movie_idx must be provided")
        
        # Find nearest neighbors
        distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=n+1)
        
        # If we used a movie_idx, exclude the movie itself from results
        if movie_idx is not None:
            # Filter out the query movie itself (which would be a perfect match)
            mask = indices[0] != movie_idx
            indices = indices[0][mask][:n]
            distances = distances[0][mask][:n]
        else:
            indices = indices[0][:n]
            distances = distances[0][:n]
        
        # Create a dataframe with results
        similar_movies = self.movies_df.iloc[indices].copy()
        similar_movies['similarity'] = 1 - distances  # Convert distance to similarity score
        
        return similar_movies
    
    def update_user_preferences(self, user_id, liked_movie_indices=None, preferred_clusters=None):
        """Update user preferences based on liked movies and explicit cluster preferences"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                'cluster_weights': np.zeros(self.num_clusters),
                'liked_movies': set()
            }
        
        # Update based on liked movies
        if liked_movie_indices is not None:
            for idx in liked_movie_indices:
                if 0 <= idx < len(self.movies_df):
                    self.user_preferences[user_id]['liked_movies'].add(idx)
                    
                    # Get cluster for this movie
                    movie_cluster = self.movies_df.iloc[idx]['cluster']
                    self.user_preferences[user_id]['cluster_weights'][movie_cluster] += 1
        
        # Update based on explicit cluster preferences
        if preferred_clusters:
            for cluster, weight in preferred_clusters.items():
                if 0 <= cluster < self.num_clusters:
                    self.user_preferences[user_id]['cluster_weights'][cluster] = weight
                    
        # Normalize weights
        cluster_sum = self.user_preferences[user_id]['cluster_weights'].sum()
        if cluster_sum > 0:
            self.user_preferences[user_id]['cluster_weights'] /= cluster_sum
            
        return self.user_preferences[user_id]
    
    def recommend_movies(self, user_id, text_query=None, top_n=10):
        """Recommend movies based on user preferences and optional text query"""
        if user_id not in self.user_preferences:
            print(f"No preferences found for user {user_id}. Creating default profile.")
            self.update_user_preferences(user_id)
        
        # Calculate base scores based on cluster preferences
        user_cluster_weights = self.user_preferences[user_id]['cluster_weights']
        
        # Get one-hot encoding of clusters
        cluster_matrix = np.eye(self.num_clusters)[self.movies_df['cluster']]
        
        # Calculate cluster similarity scores
        cluster_scores = cluster_matrix.dot(user_cluster_weights)
        
        # If text query is provided, incorporate content-based similarity
        if text_query:
            # Get query embeddings and classify it
            query_embedding = self.extract_features([text_query])[0]
            
            # Calculate text similarity with movie overviews
            text_similarities = np.zeros(len(self.movies_df))
            for i, movie_embedding in enumerate(self.movie_embeddings):
                similarity = np.dot(query_embedding, movie_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(movie_embedding) + 1e-10
                )
                text_similarities[i] = similarity
            
            # Combine both scores - text similarity is more important
            final_scores = 0.7 * text_similarities + 0.3 * cluster_scores
        else:
            final_scores = cluster_scores
        
        # Filter out movies that the user has already liked
        liked_movies = self.user_preferences[user_id]['liked_movies']
        for movie_idx in liked_movies:
            if 0 <= movie_idx < len(final_scores):
                final_scores[movie_idx] = -1  # Ensure they don't get recommended
        
        # Get top recommendations
        top_indices = np.argsort(final_scores)[::-1][:top_n]
        recommendations = self.movies_df.iloc[top_indices].copy()
        
        # Add score to recommendations
        recommendations['score'] = final_scores[top_indices]
        
        return recommendations[['title', 'overview', 'genres', 'cluster', 'score']]
    
    def get_cluster_examples(self, cluster_id, n=5):
        """Get example movies from a specific cluster"""
        cluster_movies = self.movies_df[self.movies_df['cluster'] == cluster_id]
        if len(cluster_movies) > n:
            return cluster_movies.sample(n)
        return cluster_movies


# Example usage with your dataset
def main():
    from collections import Counter
    
    # Create recommender
    recommender = MovieRecommender(num_clusters=12)  # Using 12 clusters for better categorization
    
    # Load data
    movies = recommender.load_data('data/final_movies_dataset.csv')
    
    # Create clusters and train classifier
    recommender.create_clusters()
    recommender.train_cluster_classifier()
    
    # Examine the clusters to understand movie categories
    print("\nExamining movie clusters...")
    for cluster_id in range(recommender.num_clusters):
        keywords = recommender.get_cluster_keywords(cluster_id, top_n=10)
        print(f"\nCluster {cluster_id} keywords: {', '.join([word for word, count in keywords])}")
        
        # Show some example movies from this cluster
        examples = recommender.get_cluster_examples(cluster_id, n=2)
        for _, movie in examples.iterrows():
            print(f"- {movie['title']}: {movie['overview'][:100]}...")
    
    # Test classification
    print("\nTesting overview classification...")
    test_text = "A space adventure with robots and interstellar travel."
    cluster, probs = recommender.classify_text(test_text)
    print(f"Classified text into cluster {cluster} with probability {probs[cluster]:.4f}")
    
    # Find similar movies to a query
    print("\nFinding movies similar to query...")
    query = "A family comedy about toys that come to life when humans aren't around"
    similar_movies = recommender.find_similar_movies(text=query, n=5)
    
    print(f"\nMovies similar to query: '{query}'")
    for _, movie in similar_movies.iterrows():
        print(f"- {movie['title']} (Similarity: {movie['similarity']:.4f})")
        print(f"  Overview: {movie['overview'][:100]}...")
    
    # Update user preferences
    user_id = "user1"
    print(f"\nUpdating preferences for {user_id}...")
    
    # First get some movie indices to use as liked movies
    toy_story_idx = movies[movies['title'] == 'Toy Story'].index.tolist()
    liked_indices = toy_story_idx if toy_story_idx else [0]  # Use first movie if Toy Story not found
    
    recommender.update_user_preferences(
        user_id=user_id,
        liked_movie_indices=liked_indices,
        # We can also specify preferred clusters if we know them
        preferred_clusters={0: 0.8, 1: 0.2}  # Just an example
    )
    
    # Get recommendations based on user preferences and text query
    print("\nGetting recommendations...")
    query = "I want a family movie with adventure"
    recommendations = recommender.recommend_movies(
        user_id=user_id,
        text_query=query,
        top_n=5
    )
    
    print(f"\nRecommendations based on query: '{query}'")
    for _, movie in recommendations.iterrows():
        print(f"- {movie['title']} (Score: {movie['score']:.4f}, Cluster: {movie['cluster']})")
        print(f"  Overview: {movie['overview'][:100]}...")

if __name__ == "__main__":
    main()