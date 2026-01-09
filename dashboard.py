# dashboard.py
import streamlit as st
import json
import os
import time

st.set_page_config("People Counter", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.radio("Select View", ["Viewer Panel"])

st.title("👥 People Counter - Live View")

if os.path.exists("live_data.json"):
    with open("live_data.json") as f:
        data = json.load(f)
else:
    data = {"entered": 0, "exited": 0, "current_inside": 0}

col1, col2, col3 = st.columns(3)
col1.metric("Total Entered", data["entered"])
col2.metric("Total Exited", data["exited"])
col3.metric("Currently Inside", data["current_inside"])

st.caption(f"Last updated: {data.get('last_updated', 'N/A')}")

if data["current_inside"] >= 0:
    st.success("System is Live - Count within limits")
else:
    st.error("⚠ Count anomaly detected")

time.sleep(2)
st.rerun()
