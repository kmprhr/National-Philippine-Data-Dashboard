import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="National Achievement Test Analysis",
    layout="wide"
)

# =====================================================
# DARK THEME
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #0f1117;
    color: white;
}

h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.title("🏫 Clustered School Data Visualization")

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("nat_2023-24_selected.csv")

# =====================================================
# DATA PREPARATION
# =====================================================

subject_columns = [
    "math_mps",
    "english_mps",
    "science_mps",
    "araling_panlipunan_mps",
    "filipino_mps"
]

clean_df = df.copy()

clean_df = clean_df.dropna(
    subset=subject_columns
)

# =====================================================
# OVERALL MPS
# =====================================================

clean_df["overall_mps"] = (
    clean_df[subject_columns]
    .mean(axis=1)
    .round(2)
)

# =====================================================
# STANDARDIZATION
# =====================================================

X = clean_df[subject_columns]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# K-MEANS CLUSTERING
# =====================================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

clean_df["cluster"] = clusters

# =====================================================
# PCA
# =====================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

clean_df["PCA1"] = X_pca[:, 0]
clean_df["PCA2"] = X_pca[:, 1]

# =====================================================
# AUTO LABELING
# =====================================================

def generate_cluster_label(row):

    math_sci_avg = (
        row["math_mps"] +
        row["science_mps"]
    ) / 2

    lang_avg = (
        row["english_mps"] +
        row["filipino_mps"] +
        row["araling_panlipunan_mps"]
    ) / 3

    overall = row["overall_mps"]

    if overall < 40:

        return "ALL-WEAK"

    elif overall >= 70:

        return "ALL-STRONG"

    elif math_sci_avg - lang_avg >= 10:

        return "STEM-STRONG"

    elif lang_avg - math_sci_avg >= 10:

        return "LANGUAGE-STRONG"

    else:

        return "MIXED-PERFORMANCE"

# =====================================================
# CLUSTER VISUALIZATION
# =====================================================

st.header("Visualization")

fig, ax = plt.subplots(figsize=(12, 7))

colors = {
    0: "#E24B4A",
    1: "#EF9F27",
    2: "#639922",
    3: "#378ADD"
}

cluster_labels = {
    0: "ALL-WEAK",
    1: "STEM-STRONG",
    2: "LANGUAGE-STRONG",
    3: "ALL-STRONG"
}

for cluster_id in sorted(clean_df["cluster"].unique()):

    cluster_data = clean_df[
        clean_df["cluster"] == cluster_id
    ]

    ax.scatter(
        cluster_data["math_mps"],
        cluster_data["english_mps"],
        s=cluster_data["overall_mps"] * 4,
        alpha=0.7,
        color=colors[cluster_id],
        label=cluster_labels[cluster_id]
    )

ax.set_xlim(0, 100)

ax.set_ylim(0, 100)

ax.set_xlabel("Math")

ax.set_ylabel("English")

ax.set_title(
    "Math vs English Performance Clusters"
)

legend = ax.legend(
    title="Cluster Groups",
    fontsize=10
)

ax.grid(alpha=0.3)

st.pyplot(fig)

# =====================================================
# CLUSTER SUMMARY
# =====================================================

st.header("Cluster Summary")

cluster_summary = (
    clean_df
    .groupby("cluster")
    [
        subject_columns + ["overall_mps"]
    ]
    .mean()
    .round(2)
)

cluster_summary["Cluster Type"] = (
    cluster_summary.apply(
        generate_cluster_label,
        axis=1
    )
)

cluster_summary["School Count"] = (
    clean_df["cluster"]
    .value_counts()
    .sort_index()
    .values
)

st.dataframe(cluster_summary)

# =====================================================
# CLUSTER PROFILES
# =====================================================

st.header("Cluster Profiles")

cluster_profile = (
    clean_df
    .groupby("cluster")
    [
        subject_columns + ["overall_mps"]
    ]
    .mean()
    .round(2)
)

cluster_profile["Cluster Type"] = (
    cluster_profile.apply(
        generate_cluster_label,
        axis=1
    )
)

cluster_profile["School Count"] = (
    clean_df["cluster"]
    .value_counts()
    .sort_index()
    .values
)

for cluster_id in sorted(
    clean_df["cluster"].unique()
):

    row = cluster_profile.loc[cluster_id]

    st.subheader(
        f"Cluster {cluster_id} — {row['Cluster Type']}"
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        fig, ax = plt.subplots(figsize=(8, 4))

        values = [
            row["math_mps"],
            row["english_mps"],
            row["science_mps"],
            row["araling_panlipunan_mps"],
            row["filipino_mps"]
        ]

        labels = [
            "Math",
            "English",
            "Science",
            "AP",
            "Filipino"
        ]

        ax.bar(labels, values)

        ax.set_ylim(0, 100)

        ax.set_ylabel(
            "Mean Percentage Score"
        )

        ax.set_title(
            f"Cluster {cluster_id} Subject Profile"
        )

        st.pyplot(fig)

    with col2:

        st.markdown(f"""
### Cluster Summary

**Cluster Type:**  
{row['Cluster Type']}

**School Count:**  
{int(row['School Count'])}

**Average Overall MPS:**  
{row['overall_mps']:.2f}
""")

        # =====================================================
        # RECOMMENDED INTERVENTION
        # =====================================================

        if row["Cluster Type"] == "ALL-WEAK":

            st.error("""
### Recommended Intervention

- Comprehensive academic overhaul
- Teacher training
- Infrastructure support
- Intensive remediation
- Basic literacy and numeracy programs
""")

        elif row["Cluster Type"] == "ALL-STRONG":

            st.success("""
### Recommended Intervention

- Maintain excellence
- Benchmark model schools
- Advanced enrichment programs
- Share best practices
""")

        elif row["Cluster Type"] == "STEM-STRONG":

            st.warning("""
### Recommended Intervention

- English tutoring
- Filipino comprehension support
- Reading interventions
- Writing enhancement programs
""")

        elif row["Cluster Type"] == "LANGUAGE-STRONG":

            st.warning("""
### Recommended Intervention

- Math remediation
- Science tutoring
- STEM laboratory exposure
- Quantitative reasoning support
""")

        else:

            st.info("""
### Recommended Intervention

- Balanced academic support
- Subject-specific remediation
- Performance monitoring
""")
