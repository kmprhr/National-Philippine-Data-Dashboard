import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="NAT 2023-24 Intelligence Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ─────────────────────────────────────────────────────
# FUTURISTIC THEME
# ─────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

.stApp {
    background: #050810;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(0,200,255,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(120,0,255,0.04) 0%, transparent 50%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

h1 {
    font-size: 28px !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #00c8ff, #7b2fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
}
h2 {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #c8d8f0 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-top: 2rem !important;
}
h3 { color: #8899bb !important; font-size: 14px !important; font-weight: 500 !important; }

.kpi-card {
    background: linear-gradient(135deg, #0d1526 0%, #0a1020 100%);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00c8ff, transparent);
}
.kpi-label { font-size: 11px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: #4a6080; margin-bottom: 8px; }
.kpi-value { font-size: 32px; font-weight: 700; color: #e8f4ff; font-family: 'JetBrains Mono', monospace !important; line-height: 1; }
.kpi-sub   { font-size: 12px; color: #4a6080; margin-top: 6px; }
.kpi-accent { color: #00c8ff; }
.kpi-warn   { color: #ff6b35; }
.kpi-good   { color: #00e5a0; }

.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid rgba(0,200,255,0.12);
    margin-bottom: 20px;
}
.section-badge {
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.3);
    color: #00c8ff;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 4px;
}
.section-title-text {
    font-size: 15px;
    font-weight: 600;
    color: #c8d8f0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.insight-card {
    background: #0a1020;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.insight-label { font-size: 10px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; }
.insight-text  { font-size: 13px; color: #8899bb; line-height: 1.7; }
.insight-big   { font-size: 22px; font-weight: 700; color: #e8f4ff; font-family: 'JetBrains Mono', monospace !important; }

.neo-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.2), transparent);
    margin: 2rem 0;
}

.cluster-info-card {
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.06);
    height: 100%;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# PLOTLY THEME CONSTANTS
# ─────────────────────────────────────────────────────

PLOT_BG  = "#080c18"
PAPER_BG = "#080c18"
GRID_CLR = "rgba(255,255,255,0.05)"
TEXT_CLR = "#8899bb"
FONT_FAM = "Inter"

CLUSTER_COLORS = {
    0: "#ff4466",
    1: "#ffaa00",
    2: "#00c8ff",
    3: "#00e5a0"
}

CLUSTER_LABELS = {
    0: "ALL-WEAK",
    1: "MIXED-PERFORMANCE",
    2: "AVERAGE",
    3: "ALL-STRONG"
}

RADAR_FILLS = {
    0: "rgba(255,68,102,0.12)",
    1: "rgba(255,170,0,0.12)",
    2: "rgba(0,200,255,0.12)",
    3: "rgba(0,229,160,0.12)",
}

# ─────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────

df = pd.read_csv("nat_2023-24_selected.csv")

subject_columns = [
    "math_mps",
    "english_mps",
    "science_mps",
    "araling_panlipunan_mps",
    "filipino_mps"
]

SHORT = ["Math", "English", "Science", "AP", "Filipino"]

clean_df = df.copy()
clean_df = clean_df.dropna(subset=subject_columns)

# ─────────────────────────────────────────────────────
# OVERALL MPS
# ─────────────────────────────────────────────────────

clean_df["overall_mps"] = (
    clean_df[subject_columns].mean(axis=1).round(2)
)

# ─────────────────────────────────────────────────────
# STANDARDIZATION + CLUSTERING
# ─────────────────────────────────────────────────────

X = clean_df[subject_columns]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clean_df["cluster"] = kmeans.fit_predict(X_scaled)

# Sort clusters by overall mean so 0=weakest, 3=strongest
cluster_means = clean_df.groupby("cluster")["overall_mps"].mean().sort_values()
remap = {old: new for new, old in enumerate(cluster_means.index)}
clean_df["cluster"] = clean_df["cluster"].map(remap)

# ─────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
clean_df["PCA1"] = X_pca[:, 0]
clean_df["PCA2"] = X_pca[:, 1]

# ─────────────────────────────────────────────────────
# AUTO LABELING
# ─────────────────────────────────────────────────────

def generate_cluster_label(row):
    math_sci_avg = (row["math_mps"] + row["science_mps"]) / 2
    lang_avg = (row["english_mps"] + row["filipino_mps"] + row["araling_panlipunan_mps"]) / 3
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

# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────

st.markdown("""
<div style='margin-bottom:6px;'>
    <span style='font-size:11px;letter-spacing:0.15em;color:#4a6080;text-transform:uppercase;'>
        Department of Education · Philippines
    </span>
</div>
""", unsafe_allow_html=True)

st.title("National Achievement Test Intelligence Dashboard")

st.markdown("""
<div style='font-size:13px;color:#4a6080;margin-top:4px;margin-bottom:24px;'>
    Grade 6 · School Year 2023–2024 · Subject Performance & Clustering Analysis
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────

total      = len(clean_df)
avg_mps    = clean_df["overall_mps"].mean()
critical   = (clean_df["overall_mps"] < 40).sum()
strong     = (clean_df["overall_mps"] >= 70).sum()
crit_pct   = critical / total * 100
strong_pct = strong   / total * 100

k1, k2, k3, k4, k5 = st.columns(5)

for col, label, value, sub, cls in [
    (k1, "Total Schools",    f"{total:,}",       "all schools analyzed",       "accent"),
    (k2, "Mean Overall MPS", f"{avg_mps:.1f}",   "avg across 5 subjects",      "accent"),
    (k3, "Critical Schools", f"{critical:,}",    f"{crit_pct:.1f}% · MPS < 40","warn"),
    (k4, "Strong Schools",   f"{strong:,}",      f"{strong_pct:.1f}% · MPS ≥ 70","good"),
    (k5, "Subjects Tracked", "5",                "Math · Eng · Sci · AP · Fil","accent"),
]:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value kpi-{cls}">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div class='neo-divider'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# PART 1 — CORRELATION
# ─────────────────────────────────────────────────────

st.markdown("""
<div class="section-header">
    <span class="section-badge">Part 01</span>
    <span class="section-title-text">Subject Correlation Analysis</span>
</div>
<div style='font-size:13px;color:#4a6080;margin-bottom:20px;'>
    Research Question: Are subject performances independent, or do weak students struggle across all subjects?
</div>
""", unsafe_allow_html=True)

corr_matrix = clean_df[subject_columns].corr().round(4)

col_heat, col_pair = st.columns(2)

# ── Heatmap ───────────────────────────────────────────
with col_heat:
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=SHORT, y=SHORT,
        colorscale=[
            [0.0, "#0d1526"],
            [0.3, "#0c3060"],
            [0.6, "#0055aa"],
            [0.8, "#0088dd"],
            [1.0, "#00c8ff"],
        ],
        zmin=0.55, zmax=1.0,
        text=[[f"r = {v:.4f}" for v in row] for row in corr_matrix.values],
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=TEXT_CLR, size=10),
            title=dict(text="r", font=dict(color=TEXT_CLR)),
            thickness=12, len=0.9,
        )
    ))
    fig_heat.update_layout(
        title=dict(text="Correlation Matrix — All Subjects", font=dict(size=13, color="#c8d8f0")),
        height=380,
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAM, color=TEXT_CLR),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(color=TEXT_CLR),
        yaxis=dict(color=TEXT_CLR, autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Pairwise bar ──────────────────────────────────────
with col_pair:
    pairs = []
    for i in range(len(subject_columns)):
        for j in range(i + 1, len(subject_columns)):
            pairs.append({
                "Pair": f"{SHORT[i]} ↔ {SHORT[j]}",
                "r": round(corr_matrix.values[i, j], 4)
            })
    pairs_df = pd.DataFrame(pairs).sort_values("r")

    bar_colors = [
        "#ff4466" if r < 0.68 else "#ffaa00" if r < 0.73 else "#00c8ff"
        for r in pairs_df["r"]
    ]

    fig_bar = go.Figure(go.Bar(
        x=pairs_df["r"],
        y=pairs_df["Pair"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"  r = {r:.4f}" for r in pairs_df["r"]],
        textposition="outside",
        textfont=dict(size=11, color="#8899bb"),
        hovertemplate="<b>%{y}</b><br>r = %{x:.4f}<extra></extra>"
    ))
    fig_bar.update_layout(
        title=dict(text="Pairwise Correlations (sorted)", font=dict(size=13, color="#c8d8f0")),
        height=380,
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAM, color=TEXT_CLR),
        margin=dict(l=10, r=80, t=50, b=10),
        xaxis=dict(range=[0.58, 0.88], gridcolor=GRID_CLR, color=TEXT_CLR, title="Pearson r"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", color=TEXT_CLR),
        bargap=0.3,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Insight cards ─────────────────────────────────────
strongest = pairs_df.iloc[-1]
weakest   = pairs_df.iloc[0]
avg_r     = pairs_df["r"].mean()

ci1, ci2, ci3 = st.columns(3)
with ci1:
    st.markdown(f"""
    <div class="insight-card" style="border-left:3px solid #00c8ff;">
        <div class="insight-label" style="color:#00c8ff;">Strongest Pair</div>
        <div class="insight-big">{strongest['r']:.4f}</div>
        <div class="insight-text"><b style='color:#e8f4ff'>{strongest['Pair']}</b> — nearly lockstep. Schools strong in one are almost always strong in the other.</div>
    </div>""", unsafe_allow_html=True)

with ci2:
    st.markdown(f"""
    <div class="insight-card" style="border-left:3px solid #ffaa00;">
        <div class="insight-label" style="color:#ffaa00;">Weakest Pair</div>
        <div class="insight-big">{weakest['r']:.4f}</div>
        <div class="insight-text"><b style='color:#e8f4ff'>{weakest['Pair']}</b> — still a strong positive correlation; weakest link but far from independent.</div>
    </div>""", unsafe_allow_html=True)

with ci3:
    st.markdown(f"""
    <div class="insight-card" style="border-left:3px solid #00e5a0;">
        <div class="insight-label" style="color:#00e5a0;">Key Takeaway</div>
        <div class="insight-big">{avg_r:.3f}</div>
        <div class="insight-text">Average r across all pairs. Subjects are <b style='color:#00e5a0'>NOT independent</b> — weak students struggle across all subjects.</div>
    </div>""", unsafe_allow_html=True)

# ── MPS Distribution ──────────────────────────────────
st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header" style='border-color:rgba(0,229,160,0.12);'>
    <span class="section-badge" style='color:#00e5a0;background:rgba(0,229,160,0.08);border-color:rgba(0,229,160,0.25);'>Distribution</span>
    <span class="section-title-text">Overall MPS Score Distribution</span>
</div>""", unsafe_allow_html=True)

col_dist, col_bands = st.columns([2, 1])

with col_dist:
    fig_dist = go.Figure()

    for x0, x1, color, label in [
        (0,  40,  "rgba(255,68,102,0.08)",  "Critical"),
        (40, 55,  "rgba(255,170,0,0.08)",   "Weak"),
        (55, 70,  "rgba(0,200,255,0.08)",   "Average"),
        (70, 100, "rgba(0,229,160,0.08)",   "Strong"),
    ]:
        fig_dist.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0, layer="below")
        fig_dist.add_annotation(
            x=(x0+x1)/2, y=1.05, xref="x", yref="paper",
            text=label, showarrow=False,
            font=dict(size=10, color=color.replace("0.08","0.85")),
        )

    fig_dist.add_trace(go.Histogram(
        x=clean_df["overall_mps"],
        nbinsx=50,
        marker=dict(
            color="rgba(0,200,255,0.7)",
            line=dict(color="rgba(0,200,255,0.3)", width=0.5)
        ),
        hovertemplate="MPS: %{x:.1f}<br>Schools: %{y}<extra></extra>",
        name="Schools"
    ))

    mean_val = clean_df["overall_mps"].mean()
    fig_dist.add_vline(
        x=mean_val, line_dash="dash",
        line_color="#ffaa00", line_width=2,
        annotation_text=f"Mean {mean_val:.1f}",
        annotation_font=dict(color="#ffaa00", size=11),
        annotation_position="top right"
    )

    fig_dist.update_layout(
        height=300,
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAM, color=TEXT_CLR),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        xaxis=dict(title="Overall MPS", gridcolor=GRID_CLR, color=TEXT_CLR, range=[0,100]),
        yaxis=dict(title="Number of Schools", gridcolor=GRID_CLR, color=TEXT_CLR),
        bargap=0.02,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_bands:
    total_n = len(clean_df)
    bands = [
        ("Critical", "< 40",  (clean_df["overall_mps"] < 40).sum(),                                          "#ff4466"),
        ("Weak",     "40–55", ((clean_df["overall_mps"] >= 40) & (clean_df["overall_mps"] < 55)).sum(),       "#ffaa00"),
        ("Average",  "55–70", ((clean_df["overall_mps"] >= 55) & (clean_df["overall_mps"] < 70)).sum(),       "#00c8ff"),
        ("Strong",   "70+",   (clean_df["overall_mps"] >= 70).sum(),                                          "#00e5a0"),
    ]
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    for name, rng, count, color in bands:
        pct = count / total_n * 100
        st.markdown(f"""
        <div class="insight-card" style='border-left:3px solid {color};padding:12px 16px;margin-bottom:8px;'>
            <div style='display:flex;justify-content:space-between;align-items:baseline;'>
                <span style='font-size:11px;font-weight:600;letter-spacing:0.08em;color:{color};text-transform:uppercase;'>{name} <span style='color:#4a6080;font-weight:400;'>({rng})</span></span>
                <span style='font-size:20px;font-weight:700;color:#e8f4ff;font-family:JetBrains Mono,monospace;'>{count:,}</span>
            </div>
            <div style='font-size:11px;color:#4a6080;margin:4px 0 6px;'>{pct:.1f}% of all schools</div>
            <div style='height:3px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden;'>
                <div style='width:{pct}%;height:100%;background:{color};border-radius:2px;'></div>
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div class='neo-divider'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# PART 2 — CLUSTERING
# ─────────────────────────────────────────────────────

st.markdown("""
<div class="section-header" style='border-color:rgba(170,102,255,0.15);'>
    <span class="section-badge" style='color:#aa66ff;background:rgba(170,102,255,0.08);border-color:rgba(170,102,255,0.25);'>Part 02</span>
    <span class="section-title-text">K-Means Clustering Analysis</span>
</div>
<div style='font-size:13px;color:#4a6080;margin-bottom:20px;'>
    Research Question: Can we identify distinct school types? Which at-risk patterns emerge?
</div>
""", unsafe_allow_html=True)

# ── Scatter + Radar ───────────────────────────────────
col_scatter, col_radar = st.columns(2)

with col_scatter:
    fig_scatter = go.Figure()
    for cid in sorted(clean_df["cluster"].unique()):
        cdata = clean_df[clean_df["cluster"] == cid]
        fig_scatter.add_trace(go.Scatter(
            x=cdata["math_mps"],
            y=cdata["english_mps"],
            mode="markers",
            name=f"C{cid} · {CLUSTER_LABELS[cid]}",
            marker=dict(
                color=CLUSTER_COLORS[cid],
                size=cdata["overall_mps"] / 10 + 3,
                opacity=0.65, line=dict(width=0)
            ),
            hovertemplate=(
                f"<b>Cluster {cid} — {CLUSTER_LABELS[cid]}</b><br>"
                "Math: %{x:.1f}<br>English: %{y:.1f}<extra></extra>"
            )
        ))
    fig_scatter.update_layout(
        title=dict(text="Math vs English — Cluster View", font=dict(size=13, color="#c8d8f0")),
        height=400,
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAM, color=TEXT_CLR),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(title="Math MPS", range=[0,100], gridcolor=GRID_CLR, color=TEXT_CLR),
        yaxis=dict(title="English MPS", range=[0,100], gridcolor=GRID_CLR, color=TEXT_CLR),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_radar:
    cluster_profile = (
        clean_df.groupby("cluster")[subject_columns + ["overall_mps"]]
        .mean().round(2)
    )
    categories = SHORT + [SHORT[0]]
    fig_radar = go.Figure()
    for cid in sorted(cluster_profile.index):
        vals = [cluster_profile.loc[cid, s] for s in subject_columns] + [cluster_profile.loc[cid, subject_columns[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=categories,
            fill="toself",
            name=f"C{cid} · {CLUSTER_LABELS[cid]}",
            line=dict(color=CLUSTER_COLORS[cid], width=2),
            fillcolor=RADAR_FILLS[cid],
            opacity=0.9,
        ))
    fig_radar.update_layout(
        title=dict(text="Subject Profile per Cluster", font=dict(size=13, color="#c8d8f0")),
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(visible=True, range=[0,100], gridcolor=GRID_CLR, color=TEXT_CLR, tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=GRID_CLR, color=TEXT_CLR),
        ),
        height=400,
        paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_FAM, color=TEXT_CLR),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ── Cluster Summary Table ─────────────────────────────
st.markdown("""
<div class="section-header" style='margin-top:0.5rem;'>
    <span class="section-badge">Summary</span>
    <span class="section-title-text">Cluster Summary</span>
</div>""", unsafe_allow_html=True)

cluster_summary = cluster_profile.copy()
cluster_summary["Cluster Type"] = cluster_summary.apply(generate_cluster_label, axis=1)
cluster_summary["School Count"]  = clean_df["cluster"].value_counts().sort_index().values
st.dataframe(
    cluster_summary.style.background_gradient(cmap="Blues", subset=subject_columns + ["overall_mps"]),
    use_container_width=True
)

# ── Cluster Profiles ──────────────────────────────────
st.markdown("""
<div class="section-header" style='margin-top:1.5rem;'>
    <span class="section-badge">Profiles</span>
    <span class="section-title-text">Cluster Detailed Profiles & Interventions</span>
</div>""", unsafe_allow_html=True)

INTERVENTIONS = {
    "ALL-WEAK": (
        "#ff4466", "rgba(255,68,102,0.08)", "rgba(255,68,102,0.25)",
        "Urgent Intervention Required",
        ["Comprehensive academic overhaul", "Teacher training programs",
         "Infrastructure & resource support", "Intensive remediation",
         "Basic literacy and numeracy programs"]
    ),
    "MIXED-PERFORMANCE": (
        "#ffaa00", "rgba(255,170,0,0.08)", "rgba(255,170,0,0.25)",
        "Targeted Support Needed",
        ["Balanced remediation across subjects", "Subject-specific tutoring",
         "Performance monitoring & tracking", "Peer-learning programs"]
    ),
    "STEM-STRONG": (
        "#00c8ff", "rgba(0,200,255,0.08)", "rgba(0,200,255,0.25)",
        "Language Enrichment Needed",
        ["English tutoring", "Filipino comprehension support",
         "Reading interventions", "Writing enhancement programs"]
    ),
    "LANGUAGE-STRONG": (
        "#aa66ff", "rgba(170,102,255,0.08)", "rgba(170,102,255,0.25)",
        "STEM Enrichment Needed",
        ["Math remediation", "Science tutoring",
         "STEM laboratory exposure", "Quantitative reasoning support"]
    ),
    "AVERAGE": (
        "#00c8ff", "rgba(0,200,255,0.08)", "rgba(0,200,255,0.25)",
        "Growth Acceleration",
        ["Structured enrichment programs", "Focus on weaker subject areas",
         "Teacher professional development", "Student engagement initiatives"]
    ),
    "ALL-STRONG": (
        "#00e5a0", "rgba(0,229,160,0.08)", "rgba(0,229,160,0.25)",
        "Maintain & Scale Excellence",
        ["Benchmark as model schools", "Share best practices nationwide",
         "Advanced enrichment programs", "Mentorship of struggling schools"]
    ),
}

for cid in sorted(cluster_profile.index):
    row = cluster_profile.loc[cid]
    ctype = generate_cluster_label(row)
    color, bg, border, int_title, int_items = INTERVENTIONS.get(
        ctype, ("#8899bb", "rgba(136,153,187,0.08)", "rgba(136,153,187,0.25)", "Monitoring", ["Performance monitoring"])
    )
    count = int(clean_df[clean_df["cluster"] == cid].shape[0])
    pct   = count / len(clean_df) * 100

    col_chart, col_info = st.columns([3, 2])

    with col_chart:
        vals = [row[s] for s in subject_columns]
        fig_cbar = go.Figure(go.Bar(
            x=SHORT, y=vals,
            marker=dict(
                color=[color] * 5,
                opacity=[0.4 + 0.6 * (v / 100) for v in vals],
                line=dict(width=0)
            ),
            text=[f"{v:.1f}" for v in vals],
            textposition="outside",
            textfont=dict(size=11, color=color),
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        ))
        fig_cbar.add_hline(
            y=60, line_dash="dot",
            line_color="rgba(255,255,255,0.15)", line_width=1,
            annotation_text="60 benchmark",
            annotation_font=dict(size=9, color="rgba(255,255,255,0.3)"),
            annotation_position="top right"
        )
        fig_cbar.update_layout(
            title=dict(
                text=f"Cluster {cid} — {ctype} · Overall Avg: {row['overall_mps']:.1f}",
                font=dict(size=13, color=color)
            ),
            height=300,
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
            font=dict(family=FONT_FAM, color=TEXT_CLR),
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(range=[0,110], gridcolor=GRID_CLR, color=TEXT_CLR, title="MPS"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)", color=TEXT_CLR),
        )
        st.plotly_chart(fig_cbar, use_container_width=True)

    with col_info:
        items_html = "".join([
            f"<div style='display:flex;gap:8px;margin-bottom:6px;'>"
            f"<span style='color:{color};font-size:11px;'>▸</span>"
            f"<span style='font-size:12px;color:#8899bb;'>{item}</span></div>"
            for item in int_items
        ])
        st.markdown(f"""
        <div style='background:{bg};border:1px solid {border};border-radius:12px;padding:20px;margin-top:8px;'>
            <div style='font-size:10px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:{color};margin-bottom:12px;'>
                Cluster {cid} · {ctype}
            </div>
            <div style='display:flex;gap:20px;margin-bottom:16px;'>
                <div>
                    <div style='font-size:10px;color:#4a6080;text-transform:uppercase;letter-spacing:0.08em;'>Schools</div>
                    <div style='font-size:24px;font-weight:700;color:#e8f4ff;font-family:JetBrains Mono,monospace;'>{count:,}</div>
                    <div style='font-size:11px;color:#4a6080;'>{pct:.1f}% of total</div>
                </div>
                <div>
                    <div style='font-size:10px;color:#4a6080;text-transform:uppercase;letter-spacing:0.08em;'>Avg MPS</div>
                    <div style='font-size:24px;font-weight:700;color:{color};font-family:JetBrains Mono,monospace;'>{row['overall_mps']:.1f}</div>
                    <div style='font-size:11px;color:#4a6080;'>overall mean</div>
                </div>
            </div>
            <div style='font-size:11px;font-weight:600;color:#c8d8f0;margin-bottom:10px;'>{int_title}</div>
            {items_html}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────
st.markdown("""
<div class='neo-divider'></div>
<div style='text-align:center;padding:20px 0;color:#2a3548;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;'>
    NAT Intelligence Dashboard · Grade 6 · SY 2023–2024 · Department of Education Philippines
</div>
""", unsafe_allow_html=True)