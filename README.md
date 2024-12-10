# PlaylistPulse: Neural Network-Based Song Recommender System

PlaylistPulse is a personalized song recommendation system powered by deep learning. Using a neural network-based approach, it analyzes song attributes to provide tailored music suggestions. This project demonstrates how to leverage song metadata, and lyrics embedding features to create an intelligent, adaptable recommendation engine.
Objective:

The goal of this project is to build an intelligent music recommender system that generates personalized song suggestions based on user preferences. By analyzing content-based features (song attributes and lyrics embedding features), the system aims to offer highly relevant music recommendations.
Key Features:
    
  - **Lyrics Feature Integration:** Uses both Numerical features of a song like 'dating',
     'violence', 'world/life', 'night/time', 'shake the audience',
     'family/gospel', 'romantic', 'communication', 'obscene', 'music',
     'movement/places', 'light/visual perceptions', 'family/spiritual',
     'like/girls', 'sadness', 'feelings', 'danceability', 'loudness',
     'acousticness', 'instrumentalness', 'valence', 'energy' as well as lyrics which is converted to embeddings.
  -** Real-time Personalization: **Adapt song recommendations based on recent user search.
  - **FastAPI Backend:** API server built with FastAPI, serving the recommendation model and handling user requests.
  - **React Frontend (TypeScript):** A responsive web interface that allows users to interact with the system and view personalized recommendations.
  - **Pre-trained Model and Embeddings:** The neural network model is pre-trained and deployed to provide quick, real-time recommendations.

## Architecture Overview:
1. **Model Components**:

    Song Embeddings: Captures song characteristics like genre, artist, and metadata.
    Lyrics Embeddings: Captures song lyrics.
    Audio Features: Song-specific features (e.g., tempo, energy) to enrich recommendations.
    Neural Network Layers: Dense layers process embeddings, capturing complex interactions between users and songs.

2. **Recommendation Process**:

    Song embeddings and Lyrics embeddings along with Audio Features are passed through dense layers.
    The output is the most similar songs in the database for a query song.
    The model is optimized using the Adam optimizer with binary MSE loss.

3. **Real-time API Integration**:

    FastAPI serves the model via a RESTful API.
    The API takes user inputs (e.g., song Name, Artist Name) and returns the top recommendations based on the model’s predictions.
    A React (TypeScript) web app interfaces with the FastAPI backend, displaying song recommendations to users.

## Technologies Used:

  - Backend: FastAPI, PyTorch (for model training and inference)
  - Frontend: React, TypeScript, CSS (for the user interface)
  - Model: Neural content based Filtering, Deep Learning (PyTorch)
  - Deployment: Docker (for easy containerization and deployment)

## Data Requirements:

  - Song Metadata: Genre, artist, release year, etc.
  - Lyrics : Song Lyrics fetched from genius api.
  - Audio Features : Tempo, key, energy, loudness.

## Evaluation Metrics:

  Mean Squared Error (MSE): Used for similarity score between feature embeddings.

## Project Workflow:

  - Model Training:
      song embeddings are trained.
      Audio features are used to enrich recommendations.
      The model is trained, saved as a .pth file, and includes embeddings, preprocessor, and model weights.

    - API Development:
        A FastAPI server serves the trained model and handles incoming recommendation requests.
        The API takes user input and returns personalized song recommendations.

    - Web App Interface:
        A React app (written in TypeScript) is used to display song recommendations and interact with the API.

## How to Run the Project Locally:
1. Clone the Repository:
    ```
    git clone https://github.com/CRLannister/PlaylistPulse.git
    cd PlaylistPulse
    ```
2. Set up the Backend:

    Create and activate a virtual environment.
    Install dependencies:
    ```
    poetry init
    poetry install
    ```
3. Set up the Frontend:

    Navigate to the frontend directory:
    ```      
    cd song-recommender-frontend
    ```
    Install npm dependencies:
    ```
    npm install
    ```
4. Start the Backend:
    ```
    poetry run python main.py
    ```
5. Start the Frontend:
    ```
    npm start
    ```
## Expected Outcomes:

  A fully functional song recommendation system that provides personalized suggestions based on user search and song metadata.

## Extensions and Future Enhancements:

  - Cold Start Problem: Integrate Collaboarative filtering for new users or new songs.
  - Real-Time Updates: Incorporate recent user behavior to provide up-to-date recommendations.
  - Hybrid Models: Combine collaborative filtering with content-based features (e.g., using audio features and metadata).
  - Model Fine-tuning: Adapt the model for more personalized recommendations based on evolving user preferences.

Contributing:
  Contributions are welcome! Please open an issue or submit a pull request if you'd like to contribute. Be sure to follow the project guidelines and maintain code quality.
