import os
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Crop Suitability Dashboard", layout="wide")

# --- App Title ---
st.title("Biogenic Crop Suitability Dashboard")
st.markdown("Upload a climate dataset for a South African province to assess the suitability of crops to be grown in the area.")

# --- Upload Inputs ---
st.sidebar.header("Upload Data")
crop_file = st.sidebar.file_uploader("Upload Crop Data (.xlsx)", type=["xlsx"])
climate_file = st.sidebar.file_uploader("Upload Climate Data for Province (.xlsx)", type=["xlsx"])
if uploaded_crop_file and uploaded_climate_file:
    with st.spinner("Processing data... Please wait."):
        # Read uploaded files
        crop_df = pd.read_excel(uploaded_crop_file)
        climate_df = pd.read_excel(uploaded_climate_file)

        # Process data
        processed_df = process_data(crop_df, climate_df)

        st.success("Data successfully processed.")

# --- Load Data ---
def load_crop_data(file):
    return pd.read_excel(file)

def load_climate_data(file):
    return pd.read_excel(file)

# --- Suitability Calculation ---
def calculate_suitability(climate_df, crop_df):
    results = []
    for _, crop in crop_df.iterrows():
        crop_name = crop['Crop Name']
        match_scores = []
        for _, row in climate_df.iterrows():
            score = sum([
                row['Rainfall Min'] >= crop['Rainfall Min'],
                row['Rainfall Max'] <= crop['Rainfall Max'],
                row['Temp Min'] >= crop['Temp Min'],
                row['Temp Max'] <= crop['Temp Max'],
                row['Drought Tolerance'] == crop['Drought Tolerance'],
                row['Suitable Köppen Zones'] == crop['Suitable Köppen Zones'],
                row['Soil Texture'] == crop['Soil Texture'],
                row['Drainage Preference'] == crop['Drainage Preference'],
                row['Irrigation Need'] == crop['Irrigation Need']
            ])
            results.append({
                'Crop Name': crop_name,
                'x': row['x'],
                'y': row['y'],
                'Suitability Score': score
            })
    return pd.DataFrame(results)

# --- Suitability Category ---
def categorize_score(score):
    if score >= 7:
        return 'High'
    elif score >= 4:
        return 'Moderate'
    elif score >= 0:
        return 'Low'
    else:
        return 'Unsuitable'

# --- Main Logic ---
if crop_file and climate_file:
    crop_df = load_crop_data(crop_file)
    climate_df = load_climate_data(climate_file)

    suitability_df = calculate_suitability(climate_df, crop_df)
    suitability_df['Suitability Category'] = suitability_df['Suitability Score'].apply(categorize_score)

    # --- Crop Selector ---
    selected_crops = st.multiselect("Select Crops to Compare", crop_df['Crop Name'].unique(), default=crop_df['Crop Name'].unique()[0])
    filtered_df = suitability_df[suitability_df['Crop Name'].isin(selected_crops)]

    # --- Suitability Map ---
st.subheader("Suitability Map")
fig_map = px.scatter_mapbox(
    filtered_df,
    lat="y",
    lon="x",
    color="Suitability Category",
    hover_name="Crop Name",
    mapbox_style="carto-positron",
    zoom=5,
    height=500
)
st.plotly_chart(fig_map, use_container_width=True)

    # --- Suitability Histogram ---
    st.subheader("Suitability Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=filtered_df, x="Suitability Score", hue="Crop Name", multiple="stack", bins=10, ax=ax)
    ax.set_xlabel("Suitability Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # --- Summary Table ---
    st.subheader("Summary Table")
    summary = filtered_df.groupby("Crop Name")["Suitability Score"].agg(['mean', 'min', 'max', 'count']).reset_index()
    st.dataframe(summary, use_container_width=True)

    # --- Download Button ---
    st.subheader("Download Results")
    def convert_df(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Suitability Results')
        return output.getvalue()

    st.download_button(
        label="Download Filtered Results as Excel",
        data=convert_df(filtered_df),
        file_name="suitability_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Please upload both crop and climate datasets to begin.")

# --- Footer ---
st.markdown("---")
st.markdown("© Developed by Sasol Research & Technology: Feedstock (2025)")
