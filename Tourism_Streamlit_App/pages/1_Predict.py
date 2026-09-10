import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Predict - Tourism Analytics",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Personalized Predictions & Recommendations")


# ============================================================
# Base Directory
# ============================================================

# 1_Predict.py is located at:
# Task 4/Tourism_Streamlit_App/pages/1_Predict.py
#
# Going two levels up gives:
# Task 4/

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


# ============================================================
# Load Models and Artifacts
# ============================================================

@st.cache_resource
def load_artifacts():

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        os.path.join(
            BASE_DIR,
            "tourism_cleaned_engineered.csv"
        )
    )

    # --------------------------------------------------------
    # Regression Artifacts
    # --------------------------------------------------------

    regression_model = joblib.load(
        os.path.join(
            BASE_DIR,
            "best_regression_model.pkl"
        )
    )

    regression_feature_columns = joblib.load(
        os.path.join(
            BASE_DIR,
            "regression_feature_columns.pkl"
        )
    )

    attraction_avg_rating_map = joblib.load(
        os.path.join(
            BASE_DIR,
            "attraction_avg_rating_map.pkl"
        )
    )

    # --------------------------------------------------------
    # Classification Artifacts
    # --------------------------------------------------------

    classification_model = joblib.load(
        os.path.join(
            BASE_DIR,
            "best_classification_model.pkl"
        )
    )

    classification_feature_columns = joblib.load(
        os.path.join(
            BASE_DIR,
            "classification_feature_columns.pkl"
        )
    )

    visitmode_label_map = joblib.load(
        os.path.join(
            BASE_DIR,
            "visitmode_label_map.pkl"
        )
    )

    # --------------------------------------------------------
    # Clustering Artifacts
    # --------------------------------------------------------

    clustering_model = joblib.load(
        os.path.join(
            BASE_DIR,
            "kmeans_clustering_model.pkl"
        )
    )

    clustering_scaler = joblib.load(
        os.path.join(
            BASE_DIR,
            "clustering_scaler.pkl"
        )
    )

    clustering_feature_columns = joblib.load(
        os.path.join(
            BASE_DIR,
            "clustering_feature_columns.pkl"
        )
    )

    cluster_names = joblib.load(
        os.path.join(
            BASE_DIR,
            "cluster_names.pkl"
        )
    )

    # --------------------------------------------------------
    # Recommendation Artifact
    # --------------------------------------------------------

    content_similarity_df = joblib.load(
        os.path.join(
            BASE_DIR,
            "content_similarity_matrix.pkl"
        )
    )

    # --------------------------------------------------------
    # Attraction Lookup
    #
    # Do NOT load attraction_lookup.pkl because that pickle
    # causes a pandas StringDtype compatibility error.
    #
    # Recreate the lookup directly from the CSV instead.
    # --------------------------------------------------------

    attraction_lookup = (
        df[["AttractionId", "Attraction" ]]        
        .drop_duplicates("AttractionId")
        .copy()
    )

    return (
        df,
        regression_model,
        regression_feature_columns,
        attraction_avg_rating_map,
        classification_model,
        classification_feature_columns,
        visitmode_label_map,
        clustering_model,
        clustering_scaler,
        clustering_feature_columns,
        cluster_names,
        content_similarity_df,
        attraction_lookup
    )


# ============================================================
# Load Everything
# ============================================================

(
    df,
    regression_model,
    regression_feature_columns,
    attraction_avg_rating_map,
    classification_model,
    classification_feature_columns,
    visitmode_label_map,
    clustering_model,
    clustering_scaler,
    clustering_feature_columns,
    cluster_names,
    content_similarity_df,
    attraction_lookup
) = load_artifacts()


# ============================================================
# Visit Mode Mapping
# ============================================================

inv_mode_map = {
    v: k
    for k, v in visitmode_label_map.items()
}


# ============================================================
# Feature Preparation
# ============================================================

def prepare_features(feature_columns, raw_input: dict):

    return pd.DataFrame([
        {
            c: raw_input.get(c, 0)
            for c in feature_columns
        }
    ])


# ============================================================
# End-to-End Pipeline
# ============================================================

def run_pipeline(user_id, visit_year, visit_month):

    # --------------------------------------------------------
    # Get User Data
    # --------------------------------------------------------

    user_data = df[
        df["UserId"] == user_id
    ]

    if user_data.empty:
        return None

    # --------------------------------------------------------
    # User / Attraction Information
    # --------------------------------------------------------

    target_attraction_id = (
        user_data["AttractionId"].iloc[0]
    )

    country = (
        user_data["Country"].iloc[0]
    )

    attraction_type_id = (
        df[
            df["AttractionId"] == target_attraction_id
        ]["AttractionTypeId"].iloc[0]
    )

    attr_avg = attraction_avg_rating_map.get(
        target_attraction_id,
        df["Rating"].mean()
    )

    user_visits = user_data.shape[0]

    # ========================================================
    # 1. Rating Prediction
    # ========================================================

    reg_input = {
        "VisitYear": visit_year,
        "VisitMonth": visit_month,
        "AttractionTypeId": attraction_type_id,
        "user_visit_count": user_visits,
        "attraction_avg_rating": attr_avg,
        f"Country_{country}": 1
    }

    reg_features = prepare_features(
        regression_feature_columns,
        reg_input
    )

    predicted_rating = regression_model.predict(
        reg_features
    )[0]

    # ========================================================
    # 2. Visit Mode Prediction
    # ========================================================

    clf_input = {
        **reg_input,
        "Rating": predicted_rating
    }

    clf_features = prepare_features(
        classification_feature_columns,
        clf_input
    )

    predicted_mode_id = classification_model.predict(
        clf_features
    )[0]

    predicted_mode = inv_mode_map[
        predicted_mode_id
    ]

    # ========================================================
    # 3. User Segmentation
    # ========================================================

    seg_features = pd.DataFrame([
        {
            "total_visits": user_visits,

            "unique_attractions":
                user_data["AttractionId"].nunique(),

            "avg_rating":
                user_data["Rating"].mean(),

            "unique_visit_modes":
                user_data["VisitMode"].nunique()
        }
    ])[clustering_feature_columns]

    scaled_seg_features = (
        clustering_scaler.transform(
            seg_features
        )
    )

    cluster_id = clustering_model.predict(
        scaled_seg_features
    )[0]

    segment_name = cluster_names[
        cluster_id
    ]

    # ========================================================
    # 4. Content-Based Recommendations
    # ========================================================

    visited_ids = set(
        user_data["AttractionId"]
    )

    # Use the user's first interacted attraction
    # as the recommendation reference, matching
    # the existing pipeline logic.

    scores = (
        content_similarity_df[
            target_attraction_id
        ]
        .drop(
            labels=visited_ids,
            errors="ignore"
        )
    )

    top_ids = (
        scores
        .sort_values(
            ascending=False
        )
        .head(5)
        .index
    )

    recs = (
        attraction_lookup[
            attraction_lookup["AttractionId"].isin(
                top_ids
            )
        ]["Attraction"]
        .tolist()
    )

    # ========================================================
    # Final Result
    # ========================================================

    return {
        "predicted_rating":
            round(
                float(predicted_rating),
                2
            ),

        "predicted_visit_mode":
            predicted_mode,

        "user_segment":
            segment_name,

        "recommendations":
            recs,

        "country":
            country
    }


# ============================================================
# User Interface
# ============================================================

st.markdown(
    "Enter a **User ID** from the dataset "
    "to get personalized predictions."
)


col1, col2, col3 = st.columns(3)


with col1:

    user_id = st.number_input(
        "User ID",
        min_value=1,
        value=60799,
        step=1
    )


with col2:

    visit_year = st.number_input(
        "Visit Year",
        min_value=2013,
        max_value=2026,
        value=2023
    )


with col3:

    visit_month = st.number_input(
        "Visit Month",
        min_value=1,
        max_value=12,
        value=6
    )


# ============================================================
# Run Pipeline
# ============================================================

if st.button(
    "🔮 Get Predictions",
    type="primary"
):

    result = run_pipeline(
        int(user_id),
        int(visit_year),
        int(visit_month)
    )

    if result is None:

        st.error(
            f"User {user_id} not found in the dataset. "
            "Try a different ID (e.g., 60799)."
        )

    else:

        st.success(
            "Prediction complete!"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted Rating",
            f"{result['predicted_rating']} / 5"
        )

        c2.metric(
            "Predicted Visit Mode",
            result["predicted_visit_mode"]
        )

        c3.metric(
            "User Segment",
            result["user_segment"]
        )

        st.subheader(
            "📍 Recommended Attractions"
        )

        for attraction in result["recommendations"]:

            st.write(
                f"- {attraction}"
            )