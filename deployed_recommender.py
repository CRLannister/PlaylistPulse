import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
from transformers import BertTokenizer, BertModel
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset
from tqdm import tqdm


class SimplerRecommenderNet(nn.Module):
    """Simplified Neural Network for Song Recommendation"""

    def __init__(self, input_dim, lyrics_dim=768):
        super().__init__()
        combined_dim = input_dim + lyrics_dim

        self.network = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, features, lyrics_embedding):
        combined = torch.cat([features, lyrics_embedding], dim=1)
        return self.network(combined)


class LyricProcessor:
    """Process lyrics using BERT"""

    def __init__(self, cache_file="lyrics_embeddings_cache.pkl"):
        self.cache_file = cache_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LyricProcessor using device: {self.device}")

        # Initialize BERT
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.bert.eval()  # Set to evaluation mode

    def process_lyrics_batch(self, lyrics_list):
        """Process lyrics in batches"""
        embeddings = []
        batch_size = 32

        # Ensure all lyrics are strings and clean them
        cleaned_lyrics = []
        for lyric in lyrics_list:
            if isinstance(lyric, (float, int)):
                lyric = str(lyric)
            if not lyric or lyric.isspace():
                lyric = "no lyrics available"
            cleaned_lyrics.append(lyric)

        for i in range(0, len(cleaned_lyrics), batch_size):
            batch = cleaned_lyrics[i : i + batch_size]
            batch = [str(text) for text in batch]

            try:
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.bert(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"Error processing batch: {e}")
                zero_embeddings = np.zeros((len(batch), 768))
                embeddings.extend(zero_embeddings)

        return np.array(embeddings)

    def get_cached_embeddings(self, lyrics_list):
        if os.path.exists(self.cache_file):
            print("Loading cached embeddings...")
            with open(self.cache_file, "rb") as f:
                cached_embeddings = pickle.load(f)
                if len(cached_embeddings) == len(lyrics_list):
                    return cached_embeddings
                print("Cache size mismatch. Recomputing embeddings...")

        print("Computing BERT embeddings...")
        embeddings = self.process_lyrics_batch(lyrics_list)

        print("Saving embeddings to cache...")
        with open(self.cache_file, "wb") as f:
            pickle.dump(embeddings, f)

        return embeddings


class DeployedSongRecommender:
    def __init__(
        self, songs_df, model_path="model.pth", preprocessor_path="preprocessor.pkl"
    ):
        """
        Initialize the deployed song recommender

        Args:
            songs_df: DataFrame containing song information
            model_path: Path to the saved model
            preprocessor_path: Path to the saved preprocessor
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.songs_df = songs_df
        self.model = None
        self.preprocessor = None
        self.lyric_processor = LyricProcessor()

        # Load model and preprocessor
        self.load_model(model_path, preprocessor_path)

    def load_model(self, model_path, preprocessor_path):
        """Load the trained model and preprocessor"""
        try:
            # Load preprocessor
            with open(preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)

            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            input_dim = checkpoint["input_dim"]
            self.model = SimplerRecommenderNet(input_dim=input_dim).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            print("Model and preprocessor loaded successfully!")
        except Exception as e:
            raise Exception(f"Error loading model or preprocessor: {str(e)}")

    def preprocess_data(self):
        """Preprocess the data for inference"""
        try:
            numerical_features = [
                "len",
                "danceability",
                "loudness",
                "acousticness",
                "instrumentalness",
                "valence",
                "energy",
                "age",
                "dating",
                "violence",
                "world/life",
                "night/time",
                "shake the audience",
                "family/gospel",
                "romantic",
                "communication",
                "obscene",
                "music",
                "movement/places",
                "light/visual perceptions",
                "family/spiritual",
                "like/girls",
                "sadness",
                "feelings",
            ]

            categorical_features = ["genre", "topic"]

            # Process lyrics
            lyrics_embeddings = self.lyric_processor.get_cached_embeddings(
                self.songs_df["lyrics"].fillna("").astype(str).tolist()
            )

            # Process features using loaded preprocessor
            X = self.preprocessor.transform(self.songs_df)

            # Ensure matching lengths
            min_length = min(len(X), len(lyrics_embeddings))
            X = X[:min_length]
            lyrics_embeddings = lyrics_embeddings[:min_length]

            return X, lyrics_embeddings

        except Exception as e:
            raise Exception(f"Error in preprocessing data: {str(e)}")

    def get_song_index(self, song_name=None, artist_name=None):
        """Get the index of a song based on song name and/or artist name"""
        try:
            if song_name and artist_name:
                matches = self.songs_df[
                    (self.songs_df["track_name"].str.contains(song_name, case=False))
                    & (
                        self.songs_df["artist_name"].str.contains(
                            artist_name, case=False
                        )
                    )
                ]
            elif song_name:
                matches = self.songs_df[
                    self.songs_df["track_name"].str.contains(song_name, case=False)
                ]
            else:
                raise ValueError("Please provide at least a song name")

            if len(matches) == 0:
                raise ValueError("No matching songs found")

            print("\nMatching Songs:")
            print(matches[["artist_name", "track_name", "genre"]])

            return matches.index[0]

        except Exception as e:
            raise Exception(f"Error finding song: {str(e)}")

    def recommend_similar_songs(self, song_index, top_k=5):
        """Recommend similar songs based on a given song index"""
        try:
            with torch.no_grad():
                X, lyrics_embeddings = self.preprocess_data()

                features = torch.tensor(X, dtype=torch.float32).to(self.device)
                lyrics = torch.tensor(lyrics_embeddings, dtype=torch.float32).to(
                    self.device
                )

                # Get embeddings in batches to prevent memory issues
                batch_size = 1024
                embeddings = []

                for i in range(0, len(features), batch_size):
                    batch_features = features[i : i + batch_size]
                    batch_lyrics = lyrics[i : i + batch_size]
                    batch_embeddings = (
                        self.model(batch_features, batch_lyrics).cpu().numpy()
                    )
                    embeddings.append(batch_embeddings)

                embeddings = np.vstack(embeddings)

                # Calculate similarities
                reference_embedding = embeddings[song_index]
                similarities = np.dot(embeddings, reference_embedding) / (
                    np.linalg.norm(embeddings, axis=1)
                    * np.linalg.norm(reference_embedding)
                )

                similar_indices = similarities.argsort()[::-1][1 : top_k + 1]
                recommendations = self.songs_df.iloc[similar_indices].copy()
                recommendations["similarity_score"] = similarities[similar_indices]

                return recommendations[
                    ["artist_name", "track_name", "genre", "similarity_score"]
                ]

        except Exception as e:
            raise Exception(f"Error generating recommendations: {str(e)}")


def load_and_recommend(
    songs_df_path, model_path, preprocessor_path, song_name, artist_name=None, top_k=5
):
    """Utility function to load data and get recommendations"""
    try:
        # Load songs DataFrame
        songs_df = pd.read_csv(songs_df_path)

        # Initialize recommender
        recommender = DeployedSongRecommender(
            songs_df=songs_df,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
        )

        # Get song index
        song_index = recommender.get_song_index(
            song_name=song_name, artist_name=artist_name
        )

        # Get recommendations
        recommendations = recommender.recommend_similar_songs(song_index, top_k=top_k)

        return recommendations

    except Exception as e:
        print(f"Error in recommendation process: {str(e)}")
        return None

