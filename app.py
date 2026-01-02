import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_style("whitegrid")

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Basel Traffic", layout="wide")

DATA_PATH = "basel.csv"

# =========================
# LOAD DATA
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize columns (sesuai dataset kamu)
    # day (object) -> datetime
    df["day"] = pd.to_datetime(df["day"], errors="coerce")

    # numeric columns
    for col in ["interval", "flow", "occ", "error", "speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # detid/city tetap object
    if "detid" in df.columns:
        df["detid"] = df["detid"].astype(str)
    if "city" in df.columns:
        df["city"] = df["city"].astype(str)

    # hour (0-23) dari interval detik (0..86100)
    # interval/3600 -> floor jadi jam
    df["hour"] = np.floor(df["interval"] / 3600).astype("Int64")
    df.loc[(df["hour"] < 0) | (df["hour"] > 23), "hour"] = pd.NA

    # day_type: Hari Kerja vs Akhir Pekan
    df["day_type"] = np.where(df["day"].dt.dayofweek < 5, "Hari Kerja", "Akhir Pekan")

    return df


df_raw = load_data(DATA_PATH)

# =========================
# HELPERS
# =========================
def status_kepadatan_from_flow_occ(flow_mean: float, occ_mean: float, sensitivity: str = "Sedang") -> str:
    """
    Menentukan status kepadatan (Lancar/Sedang/Padat) dari mean flow & mean occ.
    Threshold sederhana, bisa kamu ubah sesuai kebutuhan laporan.
    """
    sens_map = {
        "Rendah": 0.85,
        "Sedang": 1.00,
        "Tinggi": 1.15,
    }
    k = sens_map.get(sensitivity, 1.00)

    # Threshold heuristik (aman & masuk akal utk demo)
    # occ biasanya 0..1, flow bisa 0..1344 (dari dataset kamu)
    if (occ_mean >= 0.08 * k) or (flow_mean >= 220 * k):
        return "Padat"
    elif (occ_mean >= 0.04 * k) or (flow_mean >= 120 * k):
        return "Sedang"
    else:
        return "Lancar"


@st.cache_data(show_spinner=False)
def hourly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ringkas per jam: mean flow, mean occ, jumlah data.
    """
    tmp = df.dropna(subset=["hour", "flow", "occ"]).copy()
    tmp["hour"] = tmp["hour"].astype(int)

    out = (
        tmp.groupby("hour", as_index=False)
        .agg(
            flow_mean=("flow", "mean"),
            occ_mean=("occ", "mean"),
            jumlah_data=("flow", "count"),
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )
    return out


def line_chart_24h(df_hour: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df_hour["hour"], df_hour["flow_mean"], linewidth=2, label="flow_mean")
    ax.plot(df_hour["hour"], df_hour["occ_mean"], linewidth=2, label="occ_mean")
    ax.set_title("Grafik 24 Jam (Rata-rata Historis)")
    ax.set_xlabel("Jam (0–23)")
    ax.set_ylabel("Nilai Rata-rata")
    ax.set_xticks(range(0, 24, 1))
    ax.legend()
    return fig


# =========================
# EDA FIGURES (yang kamu minta)
# =========================
def pie_weekday_weekend_fig(df: pd.DataFrame):
    counts = df["day_type"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", startangle=90)
    ax.set_title("Pie Chart: Hari Kerja vs Akhir Pekan (Proporsi Data)")
    ax.axis("equal")
    return fig


def histogram_flow_fig(df: pd.DataFrame):
    tmp = df.dropna(subset=["flow"]).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(tmp["flow"], kde=True, bins=50, ax=ax)
    ax.set_title("Histogram Distribusi Flow")
    ax.set_xlabel("Flow")
    ax.set_ylabel("Frekuensi")
    return fig


def boxplot_flow_fig(df: pd.DataFrame):
    tmp = df.dropna(subset=["flow"]).copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(y=tmp["flow"], ax=ax)
    ax.set_title("Boxplot Flow (Outlier)")
    ax.set_ylabel("Flow")
    return fig


def interval_occ_ci_fig(df: pd.DataFrame):
    """
    Grafik Interval (jam 0-23) vs occ + CI 95%
    FIX: memastikan data 1D dengan reset_index + to_numpy().
    """
    tmp = df.dropna(subset=["hour", "occ"]).copy()
    tmp["hour"] = tmp["hour"].astype(int)
    tmp = tmp[(tmp["hour"] >= 0) & (tmp["hour"] <= 23)]

    grp = (
        tmp.groupby("hour", as_index=False)
        .agg(mean_occ=("occ", "mean"), std_occ=("occ", "std"), n=("occ", "count"))
        .sort_values("hour")
        .reset_index(drop=True)
    )

    grp["std_occ"] = grp["std_occ"].fillna(0)
    grp["se"] = grp["std_occ"] / np.sqrt(grp["n"].clip(lower=1))
    grp["ci_low"] = grp["mean_occ"] - 1.96 * grp["se"]
    grp["ci_high"] = grp["mean_occ"] + 1.96 * grp["se"]

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


# =========================
# SIDEBAR FILTERS
# =========================
st.markdown(
    """
    <div style="background:#1f4e8c;padding:22px;border-radius:14px;color:white;">
      <h1 style="margin:0;">Basel Traffic</h1>
      <p style="margin:8px 0 0 0;">
        Sistem pemantauan & estimasi lalu lintas berbasis data historis (interval 24 jam).
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# Filter bar
c1, c2, c3 = st.columns([1.2, 2.2, 2.2])
with c1:
    jam_pilih = st.slider("Pilih Jam (0–23)", 0, 23, 12)

detector_list = ["Semua Detektor"] + sorted(df_raw["detid"].dropna().unique().tolist())
with c2:
    detid_pilih = st.selectbox("Filter Detektor", detector_list, index=0)

with c3:
    jenis_hari = st.selectbox("Jenis Hari", ["Semua", "Hari Kerja", "Akhir Pekan"], index=0)

c4, _ = st.columns([2.2, 7.8])
with c4:
    sens = st.selectbox("Sensitivitas Kepadatan", ["Rendah", "Sedang", "Tinggi"], index=1)

# =========================
# APPLY FILTERS
# =========================
df_f = df_raw.copy()

if detid_pilih != "Semua Detektor":
    df_f = df_f[df_f["detid"] == detid_pilih]

if jenis_hari != "Semua":
    df_f = df_f[df_f["day_type"] == jenis_hari]

# Pastikan hour valid
df_f = df_f.dropna(subset=["hour"])
df_f["hour"] = df_f["hour"].astype(int)

# =========================
# STATUS & SUMMARY FOR SELECTED HOUR
# =========================
df_hour_selected = df_f[df_f["hour"] == jam_pilih].copy()

flow_mean_h = float(df_hour_selected["flow"].mean()) if len(df_hour_selected) else np.nan
occ_mean_h = float(df_hour_selected["occ"].mean()) if len(df_hour_selected) else np.nan
n_h = int(df_hour_selected["flow"].count()) if len(df_hour_selected) else 0

status = status_kepadatan_from_flow_occ(
    0.0 if np.isnan(flow_mean_h) else flow_mean_h,
    0.0 if np.isnan(occ_mean_h) else occ_mean_h,
    sensitivity=sens
)

st.info(f"Status Lalu Lintas: **{status}**")

# =========================
# MAIN LAYOUT (mirip contoh Melbourne)
# =========================
left, right = st.columns([1.15, 1.0])

with left:
    st.subheader("Peta / Visualisasi Lalu Lintas Basel")

    st.markdown(
        """
        <div style="background:#e9f2ff;padding:14px;border-radius:10px;">
        Dataset Basel tidak memiliki kolom koordinat (lat/lon), jadi peta tidak bisa ditampilkan.
        Sebagai pengganti, ditampilkan <b>Top Detektor</b> (flow tertinggi) pada jam yang dipilih.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    top_table = (
        df_hour_selected.dropna(subset=["flow", "occ"])
        .groupby("detid", as_index=False)
        .agg(flow_mean=("flow", "mean"), occ_mean=("occ", "mean"), n=("flow", "count"))
        .sort_values("flow_mean", ascending=False)
        .head(10)
    )
    st.dataframe(top_table, use_container_width=True, height=260)

with right:
    st.subheader("Prediksi & Analisis Lalu Lintas (Historis)")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Volume Lalu Lintas (Flow Mean)", f"{0.0 if np.isnan(flow_mean_h) else flow_mean_h:.2f}")
    m2.metric("Okupansi Jalan (Occ Mean)", f"{0.0 if np.isnan(occ_mean_h) else occ_mean_h:.4f}")
    m3.metric("Jam Dipilih", f"{jam_pilih:02d}:00")
    m4.metric("Jumlah Data (jam itu)", f"{n_h:,}")

    st.write("")
    # 24h summary (berdasarkan data yang sudah difilter)
    df_hour_sum = hourly_summary(df_f)

    fig_line = line_chart_24h(df_hour_sum)
    st.pyplot(fig_line, clear_figure=True)

    st.caption("Grafik menunjukkan rata-rata flow dan occ per jam (0–23) berdasarkan data historis (setelah filter).")

# =========================
# TABLE SUMMARY
# =========================
st.write("")
st.subheader("Tabel Ringkas per Jam")
st.dataframe(df_hour_sum, use_container_width=True, height=320)

# =========================
# EDA TABS (Pie, Histogram, Boxplot, Interval vs Occ + CI)
# =========================
st.write("")
st.header("EDA (Exploratory Data Analysis)")

tab1, tab2, tab3, tab4 = st.tabs([
    "Pie Chart (Hari Kerja vs Akhir Pekan)",
    "Histogram Flow",
    "Boxplot Flow (Outlier)",
    "Interval vs Occ + CI 95%"
])

with tab1:
    st.write(
        "Pie chart menunjukkan perbandingan persentase jumlah data pada **Hari Kerja** dan **Akhir Pekan**."
    )
    fig = pie_weekday_weekend_fig(df_f if jenis_hari == "Semua" else df_raw)  # biar pie chart tetap relevan
    st.pyplot(fig, clear_figure=True)

with tab2:
    st.write(
        "Histogram dibuat untuk melihat distribusi target **flow**. Jika condong ke kanan (right-skewed), "
        "artinya flow rendah lebih sering terjadi, sedangkan flow tinggi (macet) lebih jarang."
    )
    fig = histogram_flow_fig(df_f)
    st.pyplot(fig, clear_figure=True)

with tab3:
    st.write(
        "Boxplot memberikan ringkasan statistik visual untuk **flow**. Titik di luar whisker adalah **outlier** "
        "(kejadian ekstrem)."
    )
    fig = boxplot_flow_fig(df_f)
    st.pyplot(fig, clear_figure=True)

with tab4:
    st.write(
        "Grafik ini menunjukkan pola okupansi jalan (**occ**) rata-rata per jam (0–23). "
        "Area bayangan adalah **CI 95%** (rentang ketidakpastian rata-rata)."
    )
    fig = interval_occ_ci_fig(df_f)
    st.pyplot(fig, clear_figure=True)

st.caption("Catatan: Deploy ini menggunakan pendekatan statistik sederhana (rata-rata historis per jam). Tidak ada proses training model ML saat deploy.")
