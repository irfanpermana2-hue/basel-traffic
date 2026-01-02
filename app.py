import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Basel Traffic", layout="wide")

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
    - Rendah: lebih 'longgar' (lebih banyak dianggap lancar)
    - Sedang: default (Q1,Q2,Q3)
    - Tinggi: lebih 'ketat' (lebih cepat dianggap padat)
    """
    s = flow_series.dropna()
    if len(s) == 0:
        return None

    if sensitivity == "Rendah":
        qs = [0.35, 0.65, 0.85]
    elif sensitivity == "Tinggi":
        qs = [0.15, 0.35, 0.60]
    else:  # Sedang
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
# UI Header (mirip contoh)
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
# TOP CONTROLS (mirip Melbourne Traffic)
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

# =========================
# Apply filters
# =========================
df_f = df.copy()

if day_mode != "Semua":
    df_f = df_f[df_f["kategori_hari"] == day_mode]

if detid_selected != "Semua Detektor" and "detid" in df_f.columns:
    df_f = df_f[df_f["detid"] == detid_selected]

# =========================
# Compute summaries
# =========================
agg24 = agg_24h(df_f)
row = agg24[agg24["hour_bin"] == hour_selected]

flow_est = row["flow_mean"].iloc[0] if len(row) else np.nan
occ_est  = row["occ_mean"].iloc[0] if len(row) else np.nan
n_est    = int(row["n"].iloc[0]) if len(row) else 0

thresholds = get_thresholds(df_f["flow"], sensitivity)
status = classify_status(flow_est, thresholds)

# status message box
if status in ["Sangat Padat", "Padat"]:
    st.warning(f"Status Lalu Lintas: **{status}**")
elif status == "Sedang":
    st.info(f"Status Lalu Lintas: **{status}**")
else:
    st.success(f"Status Lalu Lintas: **{status}**")

st.markdown("---")

# =========================
# MAIN LAYOUT (kiri peta/visual, kanan kartu metrik)
# =========================
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Peta / Visualisasi Lalu Lintas Basel")

    # Kalau ada koordinat (lat/lon) bisa st.map. Saat ini Basel tidak ada -> fallback.
    has_coords = all(col in df_f.columns for col in ["lat", "lon"])  # kalau nanti kamu punya
    if has_coords:
        # contoh: tampilkan titik per detid pada jam terpilih
        det_hour = df_f[df_f["hour_bin"] == hour_selected].copy()
        det_hour = det_hour.dropna(subset=["lat", "lon"])
        st.map(det_hour.rename(columns={"lat":"latitude","lon":"longitude"}))
        st.caption("Peta menampilkan titik sensor pada jam yang dipilih.")
    else:
        st.info("Dataset Basel tidak memiliki kolom koordinat (lat/lon), jadi peta tidak bisa ditampilkan. "
                "Sebagai pengganti, ditampilkan Top Detektor (flow tertinggi) pada jam yang dipilih.")

        det_rank = agg_by_detid_at_hour(df_f, hour_selected)
        if len(det_rank) == 0:
            st.warning("Tidak ada data untuk jam/filter yang dipilih.")
        else:
            top10 = det_rank.head(10).copy()
            st.dataframe(top10, use_container_width=True, height=260)
            st.bar_chart(top10.set_index("detid")[["flow_mean"]], height=260)

    st.markdown("")

with right:
    st.subheader("Prediksi & Analisis Lalu Lintas (Historis)")

    r1, r2 = st.columns(2)
    with r1:
        st.metric("Volume Lalu Lintas (Flow Mean)", "—" if pd.isna(flow_est) else f"{flow_est:.2f}")
    with r2:
        st.metric("Okupansi Jalan (Occ Mean)", "—" if pd.isna(occ_est) else f"{occ_est:.4f}")

    r3, r4 = st.columns(2)
    with r3:
        st.metric("Jam Dipilih", f"{hour_selected}:00")
    with r4:
        st.metric("Jumlah Data (jam itu)", f"{n_est:,}")

    st.markdown("#### Grafik 24 Jam (Rata-rata Historis)")
    if len(agg24) == 0:
        st.warning("Data kosong setelah filter.")
    else:
        # pisahkan jadi 2 grafik biar occ terbaca jelas
        st.line_chart(agg24.set_index("hour_bin")[["flow_mean"]], height=210)
        st.line_chart(agg24.set_index("hour_bin")[["occ_mean"]], height=210)

    st.caption("Catatan: Estimasi ini memakai rata-rata historis pada jam yang sama (tanpa training ML saat deploy).")
