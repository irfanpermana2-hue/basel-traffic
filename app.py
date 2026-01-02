import pandas as pd
import numpy as np
import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Basel Traffic (Historis)", layout="wide")

# =========================
# FUNCTIONS
# =========================
@st.cache_data
def load_data(path: str = "basel.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Convert types safely
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df["interval"] = pd.to_numeric(df["interval"], errors="coerce")
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
    df["occ"] = pd.to_numeric(df["occ"], errors="coerce")

    # interval (seconds) -> hour
    df["hour"] = df["interval"] / 3600.0
    df["hour_bin"] = np.floor(df["hour"]).astype("Int64")

    # weekday category
    df["kategori_hari"] = df["day"].dt.weekday.apply(
        lambda x: "Akhir Pekan" if x in [5, 6] else "Hari Kerja"
    )

    return df


def drop_useless_columns(df: pd.DataFrame, null_threshold: float = 0.95) -> pd.DataFrame:
    """Drop columns that are mostly null (default >=95% null)."""
    df2 = df.copy()
    null_ratio = df2.isnull().mean()
    cols_to_drop = null_ratio[null_ratio >= null_threshold].index.tolist()
    if len(cols_to_drop) > 0:
        df2 = df2.drop(columns=cols_to_drop)
    return df2


def make_flow_thresholds(flow_series: pd.Series):
    """Auto thresholds by quantiles to classify traffic status."""
    s = flow_series.dropna()
    if len(s) == 0:
        return None
    q1, q2, q3 = s.quantile([0.25, 0.50, 0.75]).tolist()
    return q1, q2, q3


def classify_status(flow_value: float, thresholds):
    if thresholds is None or pd.isna(flow_value):
        return "Tidak tersedia"
    q1, q2, q3 = thresholds
    if flow_value <= q1:
        return "Lancar"
    elif flow_value <= q2:
        return "Sedang"
    elif flow_value <= q3:
        return "Padat"
    else:
        return "Sangat Padat"


def agg_by_hour(df_f: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean flow/occ per hour bin."""
    agg = (
        df_f.dropna(subset=["hour_bin"])
            .groupby("hour_bin", as_index=False)
            .agg(
                flow_mean=("flow", "mean"),
                occ_mean=("occ", "mean"),
                n=("flow", "size"),
            )
            .sort_values("hour_bin")
    )
    return agg


# =========================
# APP UI
# =========================
st.title("Basel Traffic")
st.caption("Deploy berbasis historis: estimasi = rata-rata flow/occ pada jam yang sama (tanpa model ML).")

# Load + auto drop mostly-null columns
df = load_data("basel.csv")
df = drop_useless_columns(df, null_threshold=0.95)

# Sidebar
st.sidebar.header("Kontrol & Filter")
mode_hari = st.sidebar.selectbox("Jenis Hari", ["Semua", "Hari Kerja", "Akhir Pekan"])

detid_list = sorted(df["detid"].dropna().unique().tolist()) if "detid" in df.columns else []
detid_selected = st.sidebar.selectbox("Detektor (detid)", ["Semua Detektor"] + detid_list)

jam_selected = st.sidebar.slider("Pilih jam (0–23)", 0, 23, 12, 1)

# Optional: filter by date range if available
use_date_filter = st.sidebar.checkbox("Filter rentang tanggal (opsional)", value=False)
date_min = df["day"].min()
date_max = df["day"].max()
if use_date_filter and pd.notna(date_min) and pd.notna(date_max):
    start_date, end_date = st.sidebar.date_input(
        "Rentang tanggal",
        value=(date_min.date(), date_max.date()),
        min_value=date_min.date(),
        max_value=date_max.date()
    )
else:
    start_date, end_date = None, None

# Apply filters
df_f = df.copy()

if mode_hari != "Semua":
    df_f = df_f[df_f["kategori_hari"] == mode_hari]

if detid_selected != "Semua Detektor" and "detid" in df_f.columns:
    df_f = df_f[df_f["detid"] == detid_selected]

if use_date_filter and start_date and end_date:
    df_f = df_f[(df_f["day"].dt.date >= start_date) & (df_f["day"].dt.date <= end_date)]

# Build hourly aggregate
agg = agg_by_hour(df_f)

# Thresholds for status based on filtered data
thresholds = make_flow_thresholds(df_f["flow"]) if "flow" in df_f.columns else None

# Pick selected hour row
row = agg[agg["hour_bin"] == jam_selected]
if len(row) == 1:
    flow_est = row["flow_mean"].iloc[0]
    occ_est = row["occ_mean"].iloc[0]
    n_est = int(row["n"].iloc[0])
else:
    flow_est, occ_est, n_est = np.nan, np.nan, 0

status = classify_status(flow_est, thresholds)

# =========================
# SUMMARY METRICS
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estimasi Flow (Mean)", "—" if pd.isna(flow_est) else f"{flow_est:.2f}")
c2.metric("Estimasi Occ (Mean)", "—" if pd.isna(occ_est) else f"{occ_est:.4f}")
c3.metric("Status Kepadatan", status)
c4.metric("Jumlah data (jam itu)", f"{n_est:,}")

st.divider()

# =========================
# MAIN PANELS
# =========================
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Grafik 24 Jam (Rata-rata Historis)")
    if len(agg) == 0:
        st.warning("Tidak ada data setelah filter diterapkan.")
    else:
        chart_df = agg.set_index("hour_bin")[["flow_mean", "occ_mean"]].copy()
        st.line_chart(chart_df, height=360)
        st.caption("Grafik menunjukkan rata-rata flow dan occ per jam (0–23) berdasarkan data historis.")

with right:
    st.subheader("Tabel Ringkas per Jam")
    if len(agg) == 0:
        st.info("Tabel tidak tersedia karena data kosong.")
    else:
        st.dataframe(
            agg.rename(columns={
                "hour_bin": "jam",
                "flow_mean": "flow_rata2",
                "occ_mean": "occ_rata2",
                "n": "jumlah_data"
            }),
            use_container_width=True,
            height=360
        )

st.divider()

# =========================
# EXTRA INSIGHTS
# =========================
st.subheader("Insight Tambahan (Lokasi Sensor)")

if "detid" in df.columns:
    df_top = df.copy()
    if mode_hari != "Semua":
        df_top = df_top[df_top["kategori_hari"] == mode_hari]
    if use_date_filter and start_date and end_date:
        df_top = df_top[(df_top["day"].dt.date >= start_date) & (df_top["day"].dt.date <= end_date)]

    top_det = (
        df_top.dropna(subset=["detid", "flow"])
              .groupby("detid", as_index=False)["flow"]
              .mean()
              .sort_values("flow", ascending=False)
              .head(10)
              .rename(columns={"flow": "flow_rata2"})
    )

    st.caption("Top 10 detid dengan rata-rata flow tertinggi (berdasarkan filter hari/tanggal).")
    st.dataframe(top_det, use_container_width=True)
else:
    st.info("Kolom detid tidak tersedia pada dataset yang dimuat.")

st.divider()

st.subheader("Contoh Data (Sample)")
show_cols = [c for c in ["day","interval","hour","hour_bin","detid","city","flow","occ","error","speed"] if c in df_f.columns]
if len(df_f) > 0:
    st.dataframe(df_f[show_cols].sample(min(15, len(df_f)), random_state=42), use_container_width=True)
else:
    st.info("Tidak ada data untuk ditampilkan.")

st.caption("Catatan: Deploy ini memakai pendekatan statistik sederhana (mean historis per jam). Tidak ada proses training model ML saat deploy.")
