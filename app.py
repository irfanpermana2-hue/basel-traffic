import pandas as pd
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Basel Traffic", layout="wide")
sns.set_style("whitegrid")

# =========================
# Load + Prep
# =========================
@st.cache_data
def load_data(path="basel.csv"):
    df = pd.read_csv(path)

    # typing
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df["interval"] = pd.to_numeric(df["interval"], errors="coerce")
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
    df["occ"]  = pd.to_numeric(df["occ"], errors="coerce")

    # interval seconds -> hour (0-24)
    df["hour"] = df["interval"] / 3600.0
    df["hour_bin"] = np.floor(df["hour"]).astype("Int64")

    # weekday category
    df["kategori_hari"] = df["day"].dt.weekday.apply(lambda x: "Akhir Pekan" if x in [5, 6] else "Hari Kerja")
    return df


def drop_useless_columns(df, null_threshold=0.95):
    null_ratio = df.isna().mean()
    drop_cols = null_ratio[null_ratio >= null_threshold].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def get_thresholds(flow_series: pd.Series, sensitivity: str):
    """
    Sensitivitas mengubah batas kategori kepadatan.
    - Rendah: lebih longgar
    - Sedang: default
    - Tinggi: lebih ketat
    """
    s = flow_series.dropna()
    if len(s) == 0:
        return None

    if sensitivity == "Rendah":
        qs = [0.35, 0.65, 0.85]
    elif sensitivity == "Tinggi":
        qs = [0.15, 0.35, 0.60]
    else:
        qs = [0.25, 0.50, 0.75]

    q1, q2, q3 = s.quantile(qs).tolist()
    return q1, q2, q3


def classify_status(flow_value, thresholds):
    if thresholds is None or pd.isna(flow_value):
        return "Tidak tersedia"
    q1, q2, q3 = thresholds
    if flow_value <= q1:
        return "Lancar"
    elif flow_value <= q2:
        return "Sedang"
    elif flow_value <= q3:
        return "Padat"
    return "Sangat Padat"


def agg_24h(df_f):
    return (
        df_f.dropna(subset=["hour_bin"])
            .groupby("hour_bin", as_index=False)
            .agg(flow_mean=("flow","mean"), occ_mean=("occ","mean"), n=("flow","size"))
            .sort_values("hour_bin")
    )


def agg_by_detid_at_hour(df_f, hour_selected: int):
    d = df_f[df_f["hour_bin"] == hour_selected].copy()
    if "detid" not in d.columns:
        return pd.DataFrame()
    out = (
        d.dropna(subset=["detid"])
         .groupby("detid", as_index=False)
         .agg(flow_mean=("flow","mean"), occ_mean=("occ","mean"), n=("flow","size"))
         .sort_values("flow_mean", ascending=False)
    )
    return out


# =========================
# Interval vs Occ + CI 95%
# =========================
def interval_occ_ci_plot(df_f: pd.DataFrame, ax=None):
    """
    Plot mean occ by time (hour 0-24) with 95% CI band.
    CI approx: mean ± 1.96 * (std/sqrt(n))
    """
    d = df_f.dropna(subset=["occ", "interval"]).copy()
    if len(d) == 0:
        return None

    d["hour"] = d["interval"] / 3600.0

    grp = (
        d.groupby("hour", as_index=False)["occ"]
         .agg(["mean", "std", "count"])
         .reset_index()
         .rename(columns={"index": "hour"})
    )
    grp["se"] = grp["std"] / np.sqrt(grp["count"].replace(0, np.nan))
    grp["ci_low"] = grp["mean"] - 1.96 * grp["se"]
    grp["ci_high"] = grp["mean"] + 1.96 * grp["se"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    ax.plot(grp["hour"], grp["mean"], label="Rata-rata occ")
    ax.fill_between(grp["hour"], grp["ci_low"], grp["ci_high"], alpha=0.2, label="CI 95%")
    ax.set_title("Grafik Interval vs Okupansi jalan (occ) dalam 24 jam")
    ax.set_xlabel("Waktu (jam) 0–24  [interval/3600]")
    ax.set_ylabel("Okupansi (occ)")
    ax.legend()
    return fig


# =========================
# UI Header
# =========================
st.markdown(
    """
    <div style="background:#1f4b8f;padding:18px 18px;border-radius:12px;margin-bottom:14px;">
        <h1 style="color:white;margin:0;">Basel Traffic</h1>
        <p style="color:#dbe8ff;margin:4px 0 0 0;">
            Sistem pemantauan & estimasi lalu lintas berbasis data historis (interval 24 jam).
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Load data
# =========================
df = load_data("basel.csv")
df = drop_useless_columns(df, null_threshold=0.95)

# =========================
# TOP CONTROLS
# =========================
cA, cB, cC, cD = st.columns([1.2, 1.3, 1.2, 1.3])

with cA:
    hour_selected = st.slider("Pilih Jam (0–23)", 0, 23, 12, 1)

with cB:
    detids = ["Semua Detektor"] + (sorted(df["detid"].dropna().unique().tolist()) if "detid" in df.columns else [])
    detid_selected = st.selectbox("Filter Detektor", detids)

with cC:
    day_mode = st.selectbox("Jenis Hari", ["Semua", "Hari Kerja", "Akhir Pekan"])

with cD:
    sensitivity = st.selectbox("Sensitivitas Kepadatan", ["Sedang", "Rendah", "Tinggi"])

# Apply filters
df_f = df.copy()

if day_mode != "Semua":
    df_f = df_f[df_f["kategori_hari"] == day_mode]

if detid_selected != "Semua Detektor" and "detid" in df_f.columns:
    df_f = df_f[df_f["detid"] == detid_selected]

# Summaries
agg24 = agg_24h(df_f)
row = agg24[agg24["hour_bin"] == hour_selected]
flow_est = row["flow_mean"].iloc[0] if len(row) else np.nan
occ_est  = row["occ_mean"].iloc[0] if len(row) else np.nan
n_est    = int(row["n"].iloc[0]) if len(row) else 0

thresholds = get_thresholds(df_f["flow"], sensitivity)
status = classify_status(flow_est, thresholds)

# Status highlight
if status in ["Sangat Padat", "Padat"]:
    st.warning(f"Status Lalu Lintas: **{status}**")
elif status == "Sedang":
    st.info(f"Status Lalu Lintas: **{status}**")
else:
    st.success(f"Status Lalu Lintas: **{status}**")

st.markdown("---")

# =========================
# MAIN LAYOUT
# =========================
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Peta / Visualisasi Lalu Lintas Basel")

    has_coords = all(col in df_f.columns for col in ["lat", "lon"])
    if has_coords:
        det_hour = df_f[df_f["hour_bin"] == hour_selected].dropna(subset=["lat", "lon"])
        st.map(det_hour.rename(columns={"lat":"latitude","lon":"longitude"}))
        st.caption("Peta menampilkan titik sensor pada jam yang dipilih.")
    else:
        st.info("Dataset Basel tidak memiliki kolom koordinat (lat/lon), jadi peta tidak bisa ditampilkan. "
                "Sebagai pengganti, ditampilkan Top Detektor (flow tertinggi) pada jam yang dipilih.")

        det_rank = agg_by_detid_at_hour(df_f, hour_selected)
        if len(det_rank) == 0:
            st.warning("Tidak ada data untuk jam/filter yang dipilih.")
        else:
            top10 = det_rank.head(10)
            st.dataframe(top10, use_container_width=True, height=260)
            st.bar_chart(top10.set_index("detid")[["flow_mean"]], height=260)

with right:
    st.subheader("Prediksi & Analisis Lalu Lintas (Historis)")

    r1, r2 = st.columns(2)
    r1.metric("Volume Lalu Lintas (Flow Mean)", "—" if pd.isna(flow_est) else f"{flow_est:.2f}")
    r2.metric("Okupansi Jalan (Occ Mean)", "—" if pd.isna(occ_est) else f"{occ_est:.4f}")

    r3, r4 = st.columns(2)
    r3.metric("Jam Dipilih", f"{hour_selected}:00")
    r4.metric("Jumlah Data (jam itu)", f"{n_est:,}")

    st.markdown("#### Grafik 24 Jam (Rata-rata Historis)")
    if len(agg24) == 0:
        st.warning("Data kosong setelah filter.")
    else:
        st.line_chart(agg24.set_index("hour_bin")[["flow_mean"]], height=220)
        st.line_chart(agg24.set_index("hour_bin")[["occ_mean"]], height=220)

    st.caption("Catatan: Estimasi memakai rata-rata historis pada jam yang sama (tanpa training ML saat deploy).")

st.markdown("---")

# =========================
# EDA SECTION (tambahan yang kamu minta)
# =========================
st.header("EDA (Exploratory Data Analysis)")

tab1, tab2, tab3, tab4 = st.tabs([
    "Pie Chart (Hari Kerja vs Akhir Pekan)",
    "Histogram Flow",
    "Boxplot Flow (Outlier)",
    "Interval vs Occ + CI 95%"
])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_style("whitegrid")

# =========================
#  Helper: label weekday/weekend
# =========================
def add_day_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df["day_type"] = np.where(df["day"].dt.dayofweek < 5, "Hari Kerja", "Akhir Pekan")
    return df

# =========================
#  1) PIE CHART: Hari Kerja vs Akhir Pekan
# =========================
def pie_weekday_weekend_fig(df: pd.DataFrame):
    df2 = add_day_type(df)

    counts = df2["day_type"].value_counts(dropna=False)
    labels = counts.index.astype(str)
    sizes = counts.values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Pie Chart: Hari Kerja vs Akhir Pekan (Proporsi Data)")
    ax.axis("equal")
    return fig

# =========================
#  2) HISTOGRAM: distribusi flow
# =========================
def histogram_flow_fig(df: pd.DataFrame):
    df2 = df.copy()
    df2["flow"] = pd.to_numeric(df2["flow"], errors="coerce")
    df2 = df2.dropna(subset=["flow"])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df2["flow"], kde=True, bins=50, ax=ax)
    ax.set_title("Histogram Distribusi Flow")
    ax.set_xlabel("Flow")
    ax.set_ylabel("Frekuensi")
    return fig

# =========================
#  3) BOXPLOT: outlier flow
# =========================
def boxplot_flow_fig(df: pd.DataFrame):
    df2 = df.copy()
    df2["flow"] = pd.to_numeric(df2["flow"], errors="coerce")
    df2 = df2.dropna(subset=["flow"])

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(y=df2["flow"], ax=ax)
    ax.set_title("Boxplot Flow (Outlier)")
    ax.set_ylabel("Flow")
    return fig

# =========================
#  4) Interval vs Occ + CI 95% (24 jam)
#     FIX ERROR fill_between: gunakan reset_index + to_numpy()
# =========================
def interval_occ_ci_fig(df: pd.DataFrame):
    df2 = df.copy()

    # pastikan numeric
    df2["interval"] = pd.to_numeric(df2["interval"], errors="coerce")
    df2["occ"] = pd.to_numeric(df2["occ"], errors="coerce")

    # buang NaN penting
    df2 = df2.dropna(subset=["interval", "occ"])

    # interval di dataset kamu = detik dalam 1 hari (0..86100)
    # ubah jadi jam 0..23 (dibulatkan)
    df2["hour"] = np.floor(df2["interval"] / 3600).astype(int)
    df2 = df2[(df2["hour"] >= 0) & (df2["hour"] <= 23)]

    # agregasi per hour
    grp = (
        df2.groupby("hour", as_index=False)
        .agg(mean_occ=("occ", "mean"),
             std_occ=("occ", "std"),
             n=("occ", "count"))
        .sort_values("hour")
        .reset_index(drop=True)
    )

    # kalau std NaN (misal n=1), set 0 biar aman
    grp["std_occ"] = grp["std_occ"].fillna(0)

    # CI 95% = mean ± 1.96 * (std/sqrt(n))
    grp["se"] = grp["std_occ"] / np.sqrt(grp["n"].clip(lower=1))
    grp["ci_low"] = grp["mean_occ"] - 1.96 * grp["se"]
    grp["ci_high"] = grp["mean_occ"] + 1.96 * grp["se"]

    # plotting (PASTI 1-D)
    x = grp["hour"].to_numpy()
    y = grp["mean_occ"].to_numpy()
    y1 = grp["ci_low"].to_numpy()
    y2 = grp["ci_high"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, y, linewidth=2, label="Rata-rata occ")
    ax.fill_between(x, y1, y2, alpha=0.2, label="CI 95%")

    ax.set_title("Grafik Interval vs Okupansi Jalan (occ) dalam 24 Jam + CI 95%")
    ax.set_xlabel("Jam (0–23) dari interval/3600")
    ax.set_ylabel("Okupansi (occ)")
    ax.set_xticks(range(0, 24, 1))
    ax.legend()
    return fig

