from fastapi.middleware.cors import CORSMiddleware                                                                                                                                                                
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional, List
import pandas as pd
from deployed_recommender import DeployedSongRecommender
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PlaylistPulse: Song Recommender API",
    description="API for getting song recommendations based on similarity",
    version="1.0.0",
)

# Add CORS middleware                                                                                                                                                                                             
app.add_middleware(                                                                                                                                                                                               
    CORSMiddleware,                                                                                                                                                                                               
    allow_origins=["http://localhost:3000",
                   "http://10.105.10.80:3000",  # Add your frontend IP
         "*"],  # React frontend URL                                                                                                                                                
    allow_credentials=True,                                                                                                                                                                                       
    allow_methods=["*"],                                                                                                                                                                                          
    allow_headers=["*"],                                                                                                                                                                                          
)                  

# Global variables for model and data
recommender = None
songs_df = None


class SongQuery(BaseModel):
    song_name: str
    artist_name: Optional[str] = None
    top_k: Optional[int] = 5


class SongRecommendation(BaseModel):
    artist_name: str
    track_name: str
    genre: str
    similarity_score: float


class RecommendationResponse(BaseModel):
    query_song: str
    query_artist: Optional[str]
    recommendations: List[SongRecommendation]


@app.on_event("startup")
async def load_model():
    """Load the model and data on startup"""
    global recommender, songs_df
    try:
        logger.info("Loading model and data...")
        # Load your dataset
        songs_df = pd.read_csv("cleaned_lyrics_data.csv")

        # Initialize recommender
        recommender = DeployedSongRecommender(
            songs_df=songs_df,
            model_path="model.pth",
            preprocessor_path="preprocessor.pkl",
        )
        logger.info("Model and data loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model and data: {str(e)}")
        raise e


@app.get("/")
async def root():
    return {"message": "Song Recommender API is running", "status": "active"}


@app.post("/recommend")
async def get_recommendations(
    song_name: str, artist_name: Optional[str] = None, top_k: int = 5
):
    try:
        if recommender is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        logger.info(f"Searching for song: {song_name} by {artist_name}")

        # Find matching songs
        if artist_name:
            matches = songs_df[
                (songs_df["track_name"].str.lower().str.contains(song_name.lower()))
                & (
                    songs_df["artist_name"]
                    .str.lower()
                    .str.contains(artist_name.lower())
                )
            ]
        else:
            matches = songs_df[
                songs_df["track_name"].str.lower().str.contains(song_name.lower())
            ]

        if len(matches) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No songs found matching '{song_name}' {f'by {artist_name}' if artist_name else ''}",
            )

        # Get the first matching song
        song_index = matches.index[0]
        matched_song = matches.iloc[0]

        logger.info(
            f"Found matching song: {matched_song['track_name']} by {matched_song['artist_name']}"
        )

        # Get recommendations
        recommendations_df = recommender.recommend_similar_songs(
            song_index=song_index, top_k=top_k
        )

        # Format response
        recommendations = []
        for _, row in recommendations_df.iterrows():
            recommendations.append(
                {
                    "artist_name": row["artist_name"],
                    "track_name": row["track_name"],
                    "genre": row["genre"],
                    "similarity_score": float(row["similarity_score"]),
                }
            )

        return {
            "query_song": matched_song["track_name"],
            "query_artist": matched_song["artist_name"],
            "recommendations": recommendations,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_songs(query: str, limit: int = 10):
    try:
        if songs_df is None:
            raise HTTPException(status_code=503, detail="Data not loaded")

        matches = songs_df[
            (songs_df["track_name"].str.lower().str.contains(query.lower()))
            | (songs_df["artist_name"].str.lower().str.contains(query.lower()))
        ]

        results = matches.head(limit)[["artist_name", "track_name", "genre"]].to_dict(
            "records"
        )

        return {"query": query, "results": results, "total_matches": len(matches)}

    except Exception as e:
        logger.error(f"Error searching songs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8032, reload=True, reload_dirs=[os.path.dirname(os.path.abspath(__file__))])

