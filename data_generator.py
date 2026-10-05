import os
import random
import pandas as pd
import numpy as np

def generate_recommender_data(num_interns=250, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 1. Course Catalog Metadata
    courses = [
        {"course_id": "ML-101", "title": "Python for Data Science & Pandas", "track": "AI & ML", "level": "Foundational", "duration_hrs": 12},
        {"course_id": "ML-102", "title": "Scikit-Learn Supervised Algorithms", "track": "AI & ML", "level": "Intermediate", "duration_hrs": 18},
        {"course_id": "ML-103", "title": "XGBoost & Ensemble Methods", "track": "AI & ML", "level": "Intermediate", "duration_hrs": 15},
        {"course_id": "ML-104", "title": "PyTorch Neural Networks & Vision", "track": "AI & ML", "level": "Advanced", "duration_hrs": 25},
        {"course_id": "ML-105", "title": "NLP with Transformers & TF-IDF", "track": "AI & ML", "level": "Advanced", "duration_hrs": 20},
        {"course_id": "DEV-201", "title": "FastAPI Microservices Architecture", "track": "Backend Dev", "level": "Foundational", "duration_hrs": 14},
        {"course_id": "DEV-202", "title": "Django REST Framework Essentials", "track": "Backend Dev", "level": "Intermediate", "duration_hrs": 20},
        {"course_id": "DEV-203", "title": "PostgreSQL Optimization & Indexing", "track": "Backend Dev", "level": "Intermediate", "duration_hrs": 16},
        {"course_id": "DEV-204", "title": "Async Worker Queues with Celery & Redis", "track": "Backend Dev", "level": "Advanced", "duration_hrs": 18},
        {"course_id": "OPS-301", "title": "Docker Containerization Fundamentals", "track": "DevOps & Cloud", "level": "Foundational", "duration_hrs": 10},
        {"course_id": "OPS-302", "title": "CI/CD Automation with GitHub Actions", "track": "DevOps & Cloud", "level": "Intermediate", "duration_hrs": 12},
        {"course_id": "OPS-303", "title": "Model Serving & Streamlit Community Cloud", "track": "DevOps & Cloud", "level": "Intermediate", "duration_hrs": 8},
        {"course_id": "BI-401", "title": "Power BI & DAX Reporting Systems", "track": "Data Analytics", "level": "Foundational", "duration_hrs": 15},
        {"course_id": "BI-402", "title": "Customer Segmentation & RFM Analytics", "track": "Data Analytics", "level": "Intermediate", "duration_hrs": 14},
    ]
    df_courses = pd.DataFrame(courses)

    # 2. Synthetic Intern Interaction History (Collaborative Filtering Rating Matrix)
    interactions = []
    course_ids = [c["course_id"] for c in courses]

    tracks = ["AI & ML", "Backend Dev", "DevOps & Cloud", "Data Analytics"]
    intern_personas = np.random.choice(tracks, size=num_interns, p=[0.40, 0.25, 0.15, 0.20])

    for i in range(1, num_interns + 1):
        intern_id = f"INT-{1000 + i}"
        favored_track = intern_personas[i - 1]

        # Intern completes between 4 to 8 courses
        num_taken = random.randint(4, 8)
        
        # Interns preferentially enroll in their preferred track
        track_courses = df_courses[df_courses['track'] == favored_track]['course_id'].tolist()
        other_courses = df_courses[df_courses['track'] != favored_track]['course_id'].tolist()

        taken_track = random.sample(track_courses, min(len(track_courses), random.randint(2, len(track_courses))))
        remaining_slots = num_taken - len(taken_track)
        taken_other = random.sample(other_courses, min(len(other_courses), max(0, remaining_slots)))

        for cid in (taken_track + taken_other):
            # Ratings: higher for favored track
            if cid in track_courses:
                rating = random.choices([4, 5, 3], weights=[0.55, 0.35, 0.10])[0]
            else:
                rating = random.choices([2, 3, 4, 5], weights=[0.25, 0.40, 0.25, 0.10])[0]

            interactions.append({
                "intern_id": intern_id,
                "course_id": cid,
                "rating": rating
            })

    df_interactions = pd.DataFrame(interactions)

    os.makedirs("data", exist_ok=True)
    df_courses.to_csv(os.path.join("data", "courses_metadata.csv"), index=False)
    df_interactions.to_csv(os.path.join("data", "intern_interactions.csv"), index=False)
    print(f"[OK] Generated {len(df_courses)} courses and {len(df_interactions)} intern interactions.")

if __name__ == "__main__":
    generate_recommender_data()