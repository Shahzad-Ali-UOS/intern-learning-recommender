# 🧭 AI Intern Learning Path Recommendation System

An intelligent recommendation engine designed for internship platforms to suggest personalized, sequenced learning paths for interns based on historical engagement patterns, domain interests, and latent skill representations.

---

## 📌 Features

- **Collaborative Filtering Engine:** Implements Matrix Factorization via **Truncated SVD** to identify latent features from intern-course interaction matrices.
- **Dynamic Learning Roadmap:** Sequentially structures recommended modules (**Foundational → Intermediate → Advanced**) to maintain logical skill progression.
- **Skill Gap Radar Analytics:** Compares completed training hours vs. projected curriculum depth across AI/ML, Backend, DevOps, and Data Analytics tracks.
- **Affinity Scoring:** Predicts user satisfaction and affinity ratings (1.0 to 5.0 scale) for uncompleted course modules.
- **Interactive UI:** Built with **Streamlit**, styled with modern dark-mode CSS cards, and powered by **Plotly** visualizations.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Machine Learning:** Scikit-learn (TruncatedSVD)
- **Data Engineering:** Pandas, NumPy
- **Dashboard & Visualizations:** Streamlit, Plotly
- **Serialization:** Joblib

---

## 📂 Project Structure

```text
intern_learning_recommender/
├── data/
│   ├── courses_metadata.csv      # Course catalog, tracks, levels, hours
│   └── intern_interactions.csv   # Historical ratings & engagement records
├── models/
│   └── recommender_artifacts.pkl # Serialized SVD model & affinity matrices
├── app.py                        # Streamlit dashboard application
├── data_generator.py             # Domain-specific dataset synthesis script
├── train.py                      # Matrix Factorization training & evaluation
├── check_intern.py               # Candidate profile audit script
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation