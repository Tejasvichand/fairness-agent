# report_viewer.py
import streamlit as st
import json

st.set_page_config(page_title="LLM Fairness Report Viewer", layout="wide")
st.title("📊 LLM Fairness Test Report Viewer")

uploaded_file = st.file_uploader("Upload LLM Fairness JSON Report", type="json")

if uploaded_file:
    report = json.load(uploaded_file)
    st.subheader("🧾 Overall Assessment")
    st.code(report.get("overall_status", "Status not found"))

    st.subheader("🧪 Giskard Findings")
    if "giskard_report" in report:
        for finding in report["giskard_report"].get("results", []):
            with st.expander(f"{finding.get('title', 'Issue')} ({finding.get('severity', 'n/a')})"):
                st.markdown(f"**Status**: `{finding.get('status', 'n/a')}`")
                st.markdown(f"**Description**: {finding.get('description', 'No description available.')}")
                if "data" in finding:
                    st.json(finding["data"])
    else:
        st.warning("Giskard report not found in uploaded file.")
