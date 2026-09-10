## System Data Flow Diagram

```mermaid
flowchart TD

    A[Tourism Dataset] --> B[Data Cleaning & EDA]

    B --> C[Feature Engineering]

    C --> D[Rating Prediction<br/>XGBoost Regression]

    C --> E[Visit Mode Prediction<br/>XGBoost Classification]

    C --> F[User Segmentation<br/>K-Means Clustering]

    C --> G[Recommendation Engine]

    G --> G1[Collaborative Filtering]
    G --> G2[Content-Based Filtering]
    G --> G3[Attraction Popularity]
    G --> G4[User Behavior]

    G1 --> H[Advanced Hybrid Recommendation]
    G2 --> H
    G3 --> H
    G4 --> H

    D --> I[End-to-End Pipeline]
    E --> I
    F --> I
    H --> I

    I --> J[Final Tourism Recommendation Output]

    J --> K[Streamlit Application]

    K --> L[Predicted Rating]
    K --> M[Predicted Visit Mode]
    K --> N[User Segment]
    K --> O[Recommended Attractions]
