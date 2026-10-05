import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AI Learning Path Optimizer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern UI Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* KPI Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(8px);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38BDF8;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    /* Learning Roadmap Step Card */
    .step-card {
        background: #0F172A;
        border-left: 4px solid #38BDF8;
        border-top: 1px solid #1E293B;
        border-right: 1px solid #1E293B;
        border-bottom: 1px solid #1E293B;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
        transition: all 0.3s ease;
    }
    .step-card:hover {
        background: #1E293B;
        border-left-color: #818CF8;
        box-shadow: 0 6px 18px rgba(56, 189, 248, 0.15);
    }
    
    /* Level Tags */
    .tag-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .tag-foundational { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .tag-intermediate { background: rgba(251, 191, 36, 0.15); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .tag-advanced { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Artifacts
# ---------------------------------------------------------
artifacts_path = os.path.join("models", "recommender_artifacts.pkl")

@st.cache_resource
def load_model_data():
    if not os.path.exists(artifacts_path):
        return None
    return joblib.load(artifacts_path)

data = load_model_data()

if data is None:
    st.error("⚠️ Model artifacts missing. Run `python train.py` first to generate models.")
    st.stop()

courses_df = data['courses_df']
rating_matrix = data['rating_matrix']
predicted_ratings = data['predicted_ratings']
variance = data['explained_variance']

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/compass--v1.png", width=64)
    st.title("System Control")
    st.caption("Matrix Factorization Collaborative Filtering Engine")
    st.markdown("---")
    
    intern_list = list(rating_matrix.index)
    selected_intern = st.selectbox("Select Active Intern:", intern_list, index=0)
    top_n = st.slider("Curate Path Length (Modules):", min_value=3, max_value=6, value=4)
    
    st.markdown("---")
    st.subheader("Model Diagnostic")
    st.markdown(f"- **Algorithm:** `Truncated SVD`")
    st.markdown(f"- **Latent Components:** `k = 6`")
    st.markdown(f"- **Explained Variance:** `{variance:.1%}`")
    st.markdown(f"- **Active Interns:** `{rating_matrix.shape[0]}`")
    st.markdown(f"- **Catalog Courses:** `{len(courses_df)}`")

# ---------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------
user_ratings = rating_matrix.loc[selected_intern]
completed_ids = user_ratings[user_ratings > 0].index.tolist()
completed_meta = courses_df[courses_df['course_id'].isin(completed_ids)].copy()
completed_meta['rating'] = completed_meta['course_id'].map(user_ratings)

# Unseen prediction sorting
all_preds = predicted_ratings.loc[selected_intern]
unseen_preds = all_preds.drop(index=completed_ids).sort_values(ascending=False)
top_rec_ids = unseen_preds.head(top_n).index.tolist()

rec_meta = courses_df[courses_df['course_id'].isin(top_rec_ids)].copy()
rec_meta['score'] = rec_meta['course_id'].map(unseen_preds)

# Calculate KPIs
total_completed_hrs = completed_meta['duration_hrs'].sum()
top_favored_track = completed_meta['track'].value_counts().idxmax() if not completed_meta.empty else "N/A"
path_recommended_hrs = rec_meta['duration_hrs'].sum()
avg_predicted_fit = rec_meta['score'].mean()

# ---------------------------------------------------------
# Header & KPI Metric Strip
# ---------------------------------------------------------
st.markdown("## 🧭 Personalized Career Path & Upskilling Engine")
st.caption(f"Real-time prescriptive skill path mapped for Intern ID: `{selected_intern}`")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Completed Modules</div>
            <div class="metric-value">{len(completed_ids)} <span style="font-size: 1rem; color: #94A3B8;">/ {len(courses_df)}</span></div>
        </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Training Hours Logged</div>
            <div class="metric-value">{total_completed_hrs} <span style="font-size: 1rem; color: #94A3B8;">hrs</span></div>
        </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Primary Persona Track</div>
            <div class="metric-value" style="font-size: 1.35rem; color: #38BDF8;">{top_favored_track}</div>
        </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Recommendation Fit</div>
            <div class="metric-value" style="color: #10B981;">{avg_predicted_fit:.2f} <span style="font-size: 1rem; color: #94A3B8;">/ 5.0</span></div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Analytics Visualizations
# ---------------------------------------------------------
col_radar, col_bars = st.columns([1.1, 1.4])

with col_radar:
    st.subheader("📊 Domain Competency Footprint")
    # Radar chart comparing current completed hours vs recommended hours per track
    all_tracks = list(courses_df['track'].unique())
    completed_hrs_track = [completed_meta[completed_meta['track'] == t]['duration_hrs'].sum() for t in all_tracks]
    rec_hrs_track = [rec_meta[rec_meta['track'] == t]['duration_hrs'].sum() for t in all_tracks]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=completed_hrs_track,
        theta=all_tracks,
        fill='toself',
        name='Completed Effort (Hrs)',
        line=dict(color='#0284C7', width=2),
        fillcolor='rgba(2, 132, 199, 0.35)'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=rec_hrs_track,
        theta=all_tracks,
        fill='toself',
        name='Recommended Path (Hrs)',
        line=dict(color='#10B981', width=2, dash='dot'),
        fillcolor='rgba(16, 185, 129, 0.25)'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=False, linecolor="#334155"),
            angularaxis=dict(tickfont=dict(size=11, color="#CBD5E1"))
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=320,
        margin=dict(l=30, r=30, t=20, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_bars:
    st.subheader("🎯 Latent Score Affinity (Unseen Modules)")
    top_chart_preds = unseen_preds.head(7).reset_index()
    top_chart_preds.columns = ['course_id', 'predicted_affinity']
    top_chart_preds = top_chart_preds.merge(courses_df[['course_id', 'title', 'track']], on='course_id')
    top_chart_preds = top_chart_preds.sort_values(by='predicted_affinity', ascending=True)

    fig_bar = px.bar(
        top_chart_preds,
        x='predicted_affinity',
        y='title',
        orientation='h',
        color='predicted_affinity',
        color_continuous_scale=['#0284C7', '#38BDF8', '#10B981'],
        range_x=[0, 5],
        hover_data=['course_id', 'track']
    )
    fig_bar.update_layout(
        height=320,
        margin=dict(l=10, r=20, t=10, b=20),
        xaxis_title="Predicted Rating (Out of 5.0)",
        yaxis_title="",
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# Step-by-Step Personalized Learning Roadmap
# ---------------------------------------------------------
st.subheader("🚀 Curated Step-by-Step Learning Pathway")
st.caption("Ordered chronologically: Foundational concepts first, followed by Intermediate & Advanced modules.")

level_hierarchy = {'Foundational': 1, 'Intermediate': 2, 'Advanced': 3}
rec_meta['level_rank'] = rec_meta['level'].map(level_hierarchy)
sorted_path = rec_meta.sort_values(by=['level_rank', 'score'], ascending=[True, False]).reset_index(drop=True)

path_cols = st.columns(len(sorted_path))

for idx, row in sorted_path.iterrows():
    level_class = f"tag-{row['level'].lower()}"
    with path_cols[idx]:
        st.markdown(f"""
            <div class="step-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: #94A3B8; font-size: 0.85rem;">MILESTONE {idx + 1}</span>
                    <span class="tag-badge {level_class}">{row['level']}</span>
                </div>
                <h4 style="margin: 0; color: #F8FAFC; font-size: 1.1rem;">{row['course_id']}</h4>
                <div style="color: #E2E8F0; font-weight: 600; font-size: 0.92rem; height: 42px; margin-top: 4px; overflow: hidden;">
                    {row['title']}
                </div>
                <div style="margin-top: 14px; font-size: 0.82rem; color: #94A3B8;">
                    <div>📂 <b>Track:</b> {row['track']}</div>
                    <div>⏱️️ <b>Effort:</b> {row['duration_hrs']} Hours</div>
                    <div style="margin-top: 6px; color: #10B981; font-weight: 600;">
                        ⭐ Affinity: {row['score']:.2f} / 5.0
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Audit Trail: Completed Courses
# ---------------------------------------------------------
with st.expander(f"🔍 Inspect Intern Historical Completions ({len(completed_meta)} Courses Recorded)"):
    if not completed_meta.empty:
        audit_table = completed_meta[['course_id', 'title', 'track', 'level', 'duration_hrs', 'rating']]
        st.dataframe(
            audit_table.rename(columns={
                'course_id': 'Course Code',
                'title': 'Module Title',
                'track': 'Track',
                'level': 'Level',
                'duration_hrs': 'Duration (Hrs)',
                'rating': 'Feedback Rating (1-5)'
            }),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No prior course completions recorded for this candidate.")