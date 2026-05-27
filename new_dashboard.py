import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

.insight-box {
    background: #1a1f2e;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #378ADD;
    margin-bottom: 16px;
    font-size: 13px;
    color: #e2e8f0;
    line-height: 1.8;
}

.finding-strong { color: #22c55e; font-weight: bold; }
.finding-weak   { color: #f97316; font-weight: bold; }
.finding-title  { color: #378ADD; font-weight: 700; font-size: 14px; margin-bottom: 8px; display: block; }

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.title("🏫 National Achievement Test — Grade 6 Analysis")

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

subject_labels = [
    "Math",
    "English",
    "Science",
    "Araling Panlipunan",
    "Filipino"
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
# =====================================================
# PART 1: CORRELATION ANALYSIS
# =====================================================
# =====================================================

st.header("📊 Part 1: Subject Correlation Analysis")

st.caption(
    "Research Question: Are subject performances independent, "
    "or do weak students struggle across all subjects?"
)

# --- Compute correlation matrix ---
corr_matrix = clean_df[subject_columns].corr().round(4)

col_heatmap, col_bars = st.columns([1, 1])

# ── Heatmap ──────────────────────────────────────────
with col_heatmap:

    st.subheader("Correlation Heatmap")

    fig_heat, ax_heat = plt.subplots(figsize=(7, 5))

    fig_heat.patch.set_facecolor("#1a1f2e")
    ax_heat.set_facecolor("#1a1f2e")

    # Draw the heatmap manually with imshow
    cmap = plt.cm.RdYlGn
    im = ax_heat.imshow(
        corr_matrix.values,
        cmap=cmap,
        vmin=0.55,
        vmax=1.0,
        aspect="auto"
    )

    # Tick labels
    ax_heat.set_xticks(range(len(subject_labels)))
    ax_heat.set_yticks(range(len(subject_labels)))
    short_labels = ["Math", "English", "Science", "AP", "Filipino"]
    ax_heat.set_xticklabels(short_labels, color="white", fontsize=10)
    ax_heat.set_yticklabels(short_labels, color="white", fontsize=10)
    plt.xticks(rotation=30, ha="right")

    # Annotate each cell with the r value
    for i in range(len(subject_columns)):
        for j in range(len(subject_columns)):
            val = corr_matrix.values[i, j]
            text_color = "black" if val > 0.75 else "white"
            ax_heat.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=10, color=text_color, fontweight="bold"
            )

    cbar = fig_heat.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("Pearson r", color="white")

    ax_heat.set_title(
        "Subject Correlation Matrix",
        color="white", fontsize=12, pad=12
    )
    ax_heat.tick_params(colors="white")
    for spine in ax_heat.spines.values():
        spine.set_edgecolor("#2e3650")

    fig_heat.tight_layout()
    st.pyplot(fig_heat)

# ── Pairwise bar chart ────────────────────────────────
with col_bars:

    st.subheader("Pairwise Correlations")

    pairs = []
    for i in range(len(subject_columns)):
        for j in range(i + 1, len(subject_columns)):
            pairs.append({
                "Pair": f"{short_labels[i]} ↔ {short_labels[j]}",
                "r": round(corr_matrix.values[i, j], 4)
            })

    pairs_df = pd.DataFrame(pairs).sort_values("r", ascending=True)

    fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
    fig_bar.patch.set_facecolor("#1a1f2e")
    ax_bar.set_facecolor("#242b3d")

    bar_colors = [
        "#E24B4A" if r < 0.68
        else "#EF9F27" if r < 0.73
        else "#22c55e"
        for r in pairs_df["r"]
    ]

    bars = ax_bar.barh(
        pairs_df["Pair"],
        pairs_df["r"],
        color=bar_colors,
        edgecolor="#0f1117",
        height=0.6
    )

    # Labels on bars
    for bar, r_val in zip(bars, pairs_df["r"]):
        ax_bar.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"r = {r_val:.4f}",
            va="center", ha="left",
            color="white", fontsize=9
        )

    ax_bar.set_xlim(0.60, 0.86)
    ax_bar.set_xlabel("Correlation (r)", color="white")
    ax_bar.set_title(
        "All Subject Pair Correlations (sorted)",
        color="white", fontsize=12, pad=12
    )
    ax_bar.tick_params(colors="white")
    ax_bar.xaxis.label.set_color("white")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#2e3650")
    ax_bar.grid(axis="x", alpha=0.2, color="white")

    fig_bar.tight_layout()
    st.pyplot(fig_bar)

# ── Insight boxes ─────────────────────────────────────

strongest_pair = pairs_df.iloc[-1]
weakest_pair   = pairs_df.iloc[0]
avg_r          = pairs_df["r"].mean()

col_i1, col_i2, col_i3 = st.columns(3)

with col_i1:
    st.markdown(
        f"""<div class="insight-box">
        <span class="finding-title">🔗 Strongest Pair</span>
        <span class="finding-strong">{strongest_pair['Pair']}</span><br>
        r = <b>{strongest_pair['r']:.4f}</b> — These two subjects move
        almost in lockstep. Schools strong in one are almost always
        strong in the other.
        </div>""",
        unsafe_allow_html=True
    )

with col_i2:
    st.markdown(
        f"""<div class="insight-box" style="border-color:#f97316">
        <span class="finding-title" style="color:#f97316">🔗 Weakest Pair</span>
        <span class="finding-weak">{weakest_pair['Pair']}</span><br>
        r = <b>{weakest_pair['r']:.4f}</b> — Still a strong positive
        correlation; the weakest link among all pairs, but far from
        independent.
        </div>""",
        unsafe_allow_html=True
    )

with col_i3:
    st.markdown(
        f"""<div class="insight-box" style="border-color:#22c55e">
        <span class="finding-title" style="color:#22c55e">📌 Key Takeaway</span>
        Average r = <b>{avg_r:.3f}</b> across all pairs.<br><br>
        Subjects are <b>NOT independent</b>. Weak students struggle
        across <em>all</em> subjects — there is no "safe" subject
        where they perform well in isolation.
        </div>""",
        unsafe_allow_html=True
    )

# ── Distribution of overall MPS ──────────────────────

st.subheader("Overall MPS Distribution")

col_dist1, col_dist2 = st.columns([2, 1])

with col_dist1:

    fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
    fig_dist.patch.set_facecolor("#1a1f2e")
    ax_dist.set_facecolor("#242b3d")

    # Histogram with colored bands
    ax_dist.axvspan(0,  40,  alpha=0.15, color="#E24B4A")
    ax_dist.axvspan(40, 55,  alpha=0.15, color="#EF9F27")
    ax_dist.axvspan(55, 70,  alpha=0.15, color="#378ADD")
    ax_dist.axvspan(70, 100, alpha=0.15, color="#22c55e")

    ax_dist.hist(
        clean_df["overall_mps"],
        bins=40,
        color="#378ADD",
        edgecolor="#0f1117",
        alpha=0.85
    )

    mean_val = clean_df["overall_mps"].mean()
    ax_dist.axvline(
        mean_val, color="#f97316",
        linewidth=2, linestyle="--",
        label=f"Mean = {mean_val:.1f}"
    )

    # Band labels
    for x, label, color in [
        (20,  "Critical\n(<40)",  "#E24B4A"),
        (47,  "Weak\n(40–55)",    "#EF9F27"),
        (62,  "Average\n(55–70)", "#a0aec0"),
        (82,  "Strong\n(70+)",    "#22c55e"),
    ]:
        ax_dist.text(
            x, ax_dist.get_ylim()[1] * 0.92 if ax_dist.get_ylim()[1] > 0 else 50,
            label, ha="center", va="top",
            color=color, fontsize=9, fontweight="bold"
        )

    ax_dist.set_xlabel("Overall MPS", color="white")
    ax_dist.set_ylabel("Number of Schools", color="white")
    ax_dist.set_title(
        "Distribution of Overall MPS Across All Schools",
        color="white", fontsize=12
    )
    ax_dist.tick_params(colors="white")
    ax_dist.legend(
        facecolor="#1a1f2e", labelcolor="white",
        edgecolor="#2e3650"
    )
    for spine in ax_dist.spines.values():
        spine.set_edgecolor("#2e3650")

    fig_dist.tight_layout()
    st.pyplot(fig_dist)

with col_dist2:

    # Band counts
    bands = [
        ("Critical (<40)",  (clean_df["overall_mps"] < 40).sum(),          "#E24B4A"),
        ("Weak (40–55)",    ((clean_df["overall_mps"] >= 40) & (clean_df["overall_mps"] < 55)).sum(), "#EF9F27"),
        ("Average (55–70)", ((clean_df["overall_mps"] >= 55) & (clean_df["overall_mps"] < 70)).sum(), "#378ADD"),
        ("Strong (70+)",    (clean_df["overall_mps"] >= 70).sum(),          "#22c55e"),
    ]

    total = len(clean_df)
    st.markdown("<br>", unsafe_allow_html=True)
    for label, count, color in bands:
        pct = count / total * 100
        st.markdown(
            f"""<div style="background:#1a1f2e;border-radius:8px;
                padding:10px 14px;margin-bottom:8px;
                border-left:4px solid {color}">
                <span style="color:{color};font-weight:700">{label}</span><br>
                <span style="font-size:20px;font-weight:700;color:white">{count:,}</span>
                <span style="color:#a0aec0;font-size:12px"> schools ({pct:.1f}%)</span>
            </div>""",
            unsafe_allow_html=True
        )

st.divider()


# =====================================================
# =====================================================
# PART 2: CLUSTERING
# =====================================================
# =====================================================

st.header("🔵 Part 2: Clustering Analysis")

# =====================================================
# CLUSTER VISUALIZATION
# =====================================================

st.subheader("Cluster Scatter Plot")

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

st.subheader("Cluster Summary")

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

st.subheader("Cluster Profiles")

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