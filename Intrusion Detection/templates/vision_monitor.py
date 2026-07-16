import os
import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# --- CONFIG ---
st.set_page_config(page_title="Vision Monitor Dashboard", layout="wide")
STATIC_ROOT = "Static_Storage"


@st.cache_data(ttl=5)
def load_data(cam_name: str) -> pd.DataFrame:
    log_path = os.path.join(STATIC_ROOT, cam_name, "Logs", "activity_log.json")
    if not os.path.exists(log_path):
        return pd.DataFrame()

    rows = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame(rows)

    df = pd.DataFrame(rows)
    # Defensive: ensure expected columns exist
    for col in ("in", "out"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def main():
    st.title("🏙️ Real-Time Smart City Monitoring")

    # Sidebar
    cam_choice = st.sidebar.selectbox("Select Camera", sorted(os.listdir(STATIC_ROOT)) if os.path.exists(STATIC_ROOT) else ["Dublin", "NYC"])
    refresh_rate = st.sidebar.slider("Refresh Interval (s)", 2, 60, 5)
    st.sidebar.caption("Automatic refresh uses optional package `streamlit-autorefresh`.")

    # Try to enable auto-refresh if package is present
    auto_refresh_enabled = False
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=refresh_rate * 1000, key=f"autorefresh-{cam_choice}")
        auto_refresh_enabled = True
    except Exception:
        auto_refresh_enabled = False

    df = load_data(cam_choice)

    # Metrics
    col1, col2, col3 = st.columns(3)
    if not df.empty:
        total_tracked = len(df)
        if "in" in df.columns and "out" in df.columns:
            df["duration"] = (df["out"] - df["in"]).dt.total_seconds()
            avg_stay = df["duration"].mean()
            avg_stay_str = f"{avg_stay:.1f}"
        else:
            avg_stay_str = "N/A"

        col1.metric("Total Detections", total_tracked)
        col2.metric("Avg. Stay (s)", avg_stay_str)
        col3.metric("Current Status", "Active", delta="OK")
    else:
        st.warning("No logs found for this camera yet.")

    # Tabs for visuals and captures
    tab1, tab2 = st.tabs(["Analytics", "Recent Captures"])

    with tab1:
        if not df.empty and "in" in df.columns:
            df["hour"] = df["in"].dt.hour
            fig = px.histogram(df, x="hour", title="Hourly Activity Trend", nbins=24)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Recent Activity Log")
            st.dataframe(df.sort_values(by="in", ascending=False).head(10), use_container_width=True)
        else:
            st.info("Not enough data to show analytics.")

    with tab2:
        cap_path = os.path.join(STATIC_ROOT, cam_choice, "Captures")
        if os.path.exists(cap_path):
            imgs = sorted([f for f in os.listdir(cap_path) if os.path.isfile(os.path.join(cap_path, f))], reverse=True)[:6]
            if imgs:
                cols = st.columns(3)
                for idx, img_name in enumerate(imgs):
                    with cols[idx % 3]:
                        st.image(os.path.join(cap_path, img_name), caption=img_name)
            else:
                st.info("No capture images found yet.")
        else:
            st.info("Captures folder not found for this camera.")

    # Manual refresh fallback
    if not auto_refresh_enabled:
        if st.button("Manual Refresh"):
            st.experimental_rerun()

    st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
