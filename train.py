import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

def train_recommender():
    courses_path = os.path.join("data", "courses_metadata.csv")
    interactions_path = os.path.join("data", "intern_interactions.csv")

    if not (os.path.exists(courses_path) and os.path.exists(interactions_path)):
        from data_generator import generate_recommender_data
        generate_recommender_data()

    df_courses = pd.read_csv(courses_path)
    df_interactions = pd.read_csv(interactions_path)

    # 1. Pivot to User-Item Matrix (Interns x Courses)
    rating_matrix = df_interactions.pivot(index='intern_id', columns='course_id', values='rating').fillna(0).astype(float)
    
    # Calculate non-zero mean per intern
    mask = rating_matrix > 0
    intern_means = rating_matrix.replace(0, np.nan).mean(axis=1).fillna(3.0)

    # Vectorized mean centering on observed ratings (maintains 2D float DataFrame)
    matrix_centered = rating_matrix.copy()
    for intern_id in rating_matrix.index:
        mean_val = intern_means[intern_id]
        matrix_centered.loc[intern_id] = np.where(
            rating_matrix.loc[intern_id] > 0,
            rating_matrix.loc[intern_id] - mean_val,
            0.0
        )

    # Ensure pure float64 array for TruncatedSVD
    matrix_centered = matrix_centered.astype(np.float64)

    # 2. Matrix Factorization via TruncatedSVD
    n_components = min(6, rating_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_user_matrix = svd.fit_transform(matrix_centered.values)
    latent_item_matrix = svd.components_

    # 3. Reconstruct Affinity Matrix
    reconstructed_matrix = np.dot(latent_user_matrix, latent_item_matrix)
    predicted_ratings_df = pd.DataFrame(
        reconstructed_matrix, 
        index=rating_matrix.index, 
        columns=rating_matrix.columns
    )

    # Add intern baseline means back
    for intern_id in predicted_ratings_df.index:
        predicted_ratings_df.loc[intern_id] += intern_means[intern_id]

    explained_variance = float(svd.explained_variance_ratio_.sum())

    print("=" * 55)
    print("   COLLABORATIVE FILTERING MATRIX FACTORIZATION")
    print("=" * 55)
    print(f"User-Item Matrix Size:       {rating_matrix.shape[0]} interns x {rating_matrix.shape[1]} courses")
    print(f"Latent Components (k):       {n_components}")
    print(f"Total Explained Variance:    {explained_variance:.2%}")
    print("=" * 55)

    os.makedirs("models", exist_ok=True)
    artifacts = {
        'svd_model': svd,
        'rating_matrix': rating_matrix,
        'predicted_ratings': predicted_ratings_df,
        'courses_df': df_courses,
        'explained_variance': explained_variance
    }
    artifact_path = os.path.join("models", "recommender_artifacts.pkl")
    joblib.dump(artifacts, artifact_path)
    print(f"[OK] Recommender artifacts saved -> {artifact_path}")

if __name__ == "__main__":
    train_recommender()