import os
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO, StringIO

st.set_page_config(page_title="Crop Suitability Assessment Tool (CSAT)", layout="wide")

st.title("Crop Suitability Assessment Tool (CSAT)")
st.markdown("⚠️ Disclaimer: This tool provides an initial crop suitability estimate based on your data. Results are indicative. Ensure your data is accurate for best outcomes.")
st.warning("Ensure your Land and Climate Data Excel files follow the required format and naming convention: '<Province abbreviation>_coordinates.xlsx'.")

# Upload section
st.sidebar.header("Upload Data")
crop_file = st.sidebar.file_uploader("Upload Crop Data (.xlsx)", type=["xlsx"])
climate_files = st.sidebar.file_uploader("Upload Land and Climate Data for Province (.xlsx)", type=["xlsx"], accept_multiple_files=True)

# --- Load Data ---
@st.cache_data
def load_crop_data(file):
    return pd.read_excel(file)

@st.cache_data
def load_climate_data(files):
    df_list = []
    for f in files:
        temp_df = pd.read_excel(f)
        temp_df["source_file"] = f.name   # Add this line
        df_list.append(temp_df)
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

# --- Helper function for multi-match ---
def is_multi_match(crop_val, land_val):
    if pd.isna(crop_val) or pd.isna(land_val):
        return False
    try:
        crop_set = set(str(crop_val).replace(" ", "").split(","))
        land_set = set(str(land_val).replace(" ", "").split(","))
        return bool(crop_set & land_set)
    except Exception:
        return False

# --- failure checker ---
def check_failures(row, crop):
    failures = []

    # Temperature: crop must fit within area's range
    if crop['Temp Min'] < row['Temp Min']:
        failures.append('Temp Min')

    # Fail if crop requires lower maximum temp than area allows
    if crop['Temp Max'] > row['Temp Max']:
        failures.append('Temp Max')

    # Fail if crop requires more rainfall than area provides
    if crop['Rainfall Min'] > row['Rainfall Min']:
        failures.append('Rainfall Min')

    if crop['Rainfall Max'] > row['Rainfall Max']:
        failures.append('Rainfall Max')

    # Drought tolerance: strict equality
    if str(row['Drought Tolerance']).strip() != str(crop['Drought Tolerance']).strip():
        failures.append('Drought Tolerance')

    # Köppen Zones, Soil Texture, Drainage Preference
    if not is_multi_match(crop['Suitable Köppen Zones'], row['Suitable Köppen Zones']):
        failures.append('Suitable Köppen Zones')
    if not is_multi_match(crop['Soil Texture'], row['Soil Texture']):
        failures.append('Soil Texture')
    if not is_multi_match(crop['Drainage Preference'], row['Drainage Preference']):
        failures.append('Drainage Preference')

    # Irrigation Need: strict equality (matches your definition)
    if str(row['Irrigation Need']).strip().lower() != str(crop['Irrigation Need']).strip().lower():
        failures.append('Irrigation Need')

    return ', '.join(failures) if failures else 'None'


# --- suitability calculation ---
def calculate_suitability(climate_df, crop_df):
    results = []
    for _, crop in crop_df.iterrows():
        crop_name = crop['Crop Name']
        for _, row in climate_df.iterrows():
            score = sum([
                    crop['Rainfall Min'] <= row['Rainfall Min'],   # match if crop min <= area min
                    crop['Rainfall Max'] <= row['Rainfall Max'],   # match if crop max <= area max
                    crop['Temp Min'] <= row['Temp Min'],           # match if crop min <= area min
                    crop['Temp Max'] <= row['Temp Max'],           # match if crop max <= area max
                    str(row['Drought Tolerance']).strip() == str(crop['Drought Tolerance']).strip(),
                    is_multi_match(crop['Suitable Köppen Zones'], row['Suitable Köppen Zones']),
                    is_multi_match(crop['Soil Texture'], row['Soil Texture']),
                    is_multi_match(crop['Drainage Preference'], row['Drainage Preference']),
                    str(row['Irrigation Need']).strip().lower() == str(crop['Irrigation Need']).strip().lower()])


            results.append({
                'Crop Name': crop_name,
                'x': row.get('x', None),
                'y': row.get('y', None),
                'Rainfall Min': row['Rainfall Min'],
                'Rainfall Max': row['Rainfall Max'],
                'Temp Min': row['Temp Min'],
                'Temp Max': row['Temp Max'],
                'Suitability Score': score,
                'Failure Reasons': check_failures(row, crop)
            })
    return pd.DataFrame(results)

def categorize_score(score):
    if score >= 7:
        return 'High'
    elif score >= 4:
        return 'Moderate'
    elif score >= 1:
        return 'Low'
    else:
        return 'Unsuitable'

# ---------------------- MAIN LOGIC ----------------------
if crop_file and climate_files:
    with st.spinner("Loading & processing data..."):
        crop_df = load_crop_data(crop_file)
        climate_df = load_climate_data(climate_files)

        # Handle fallow land field
        if "Fallow land area" in climate_df.columns:
            climate_df.rename(columns={"Fallow land area": "area_ha"}, inplace=True)
        climate_df["area_ha"] = pd.to_numeric(climate_df.get("area_ha", 0), errors="coerce").fillna(0)

    st.success("✅ Data loaded successfully.")

    # Sidebar selection
    selected_provinces = st.sidebar.multiselect("Select Provinces", climate_df['source_file'].unique())
    selected_crops = st.sidebar.multiselect("Select Crops", crop_df['Crop Name'].unique())

    if not selected_provinces or not selected_crops:
        st.info("⬅️ Select provinces and crops to continue")
        st.stop()

    # Filter based on user selection
    filtered_climate = climate_df[climate_df['source_file'].isin(selected_provinces)]
    filtered_crop = crop_df[crop_df['Crop Name'].isin(selected_crops)]

    # Calculate suitability
    suitability_df = calculate_suitability(filtered_climate, filtered_crop)
    suitability_df = suitability_df.merge(
        filtered_climate[['x', 'y', 'area_ha', 'source_file']], 
        on=['x', 'y'], how='left'
    )
    suitability_df['Suitability Category'] = suitability_df['Suitability Score'].apply(categorize_score)


    # ---------------------- MAP ----------------------
    st.subheader("📍 Suitability Map")

    plot_df = suitability_df.copy()
    plot_df['Score'] = plot_df['Suitability Score']
    plot_df.loc[plot_df['Suitability Category']=="Unsuitable", 'Score'] = -1

    fig = px.scatter_mapbox(
        plot_df, lat="y", lon="x",
        color="Score",
        color_continuous_scale=["lightgrey","red","orange","green"],
        range_color=[-1,9],
        hover_data=["Crop Name","Suitability Score","Failure Reasons","area_ha"],
        mapbox_style="open-street-map", zoom=5, height=500
    )
    st.plotly_chart(fig, use_container_width=True)


    # ---------------------- Pie Chart ----------------------
    selected_crop = st.selectbox("Select crop for breakdown", suitability_df['Crop Name'].unique())
    crop_df_sel = plot_df[plot_df['Crop Name']==selected_crop]

    st.plotly_chart(
        px.pie(crop_df_sel, names="Suitability Category", title=f"Suitability Distribution: {selected_crop}"),
        use_container_width=True
    )


    # ---------------------- Top Crop Areas ----------------------
    high_df = suitability_df[suitability_df["Suitability Category"]=="High"]
    top_area = high_df.groupby("Crop Name")["area_ha"].sum().reset_index().sort_values(by="area_ha", ascending=False).head(10)

    st.subheader(" Top Suitable Crops by Area (ha)")
    st.plotly_chart(px.bar(top_area, x="Crop Name", y="area_ha"), use_container_width=True)


    # ---------------------- Provincial Summary ----------------------
    st.subheader(" Provincial Suitability Summary")

    summary = suitability_df.groupby(["source_file", "Crop Name"]).agg(
        Avg_Score=("Suitability Score","mean"),
        High=("Suitability Category", lambda x: (x=="High").mean()*100),
        Moderate=("Suitability Category", lambda x: (x=="Moderate").mean()*100),
        Low=("Suitability Category", lambda x: (x=="Low").mean()*100),
        Main_Failure=("Failure Reasons", lambda x: x.value_counts().idxmax() if any(x != 'None') else 'None')
    ).reset_index()

    st.dataframe(summary, use_container_width=True)

else:
    st.info(" Upload datasets to begin.")


# --- Footer ---
st.markdown("---")
st.markdown("© Developed by Sasol Research & Technology: Feedstock (2025)")




