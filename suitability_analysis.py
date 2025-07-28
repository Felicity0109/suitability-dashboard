import os
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Crop Suitability Assessment Tool (CSAT)", layout="wide")

# --- App Title ---
st.title("Crop Suitability Assessment Tool (CSAT)")
st.markdown("⚠️ Disclaimer: This tool provides an initial crop suitability estimate based on your data. Results are indicative. Ensure your data is accurate for best outcomes.")
st.warning("Ensure your input Excel files follow the required format and naming convention: '<Province abbreviation>_coordinates.xlsx'.")

# --- Upload Inputs ---
st.sidebar.header("Upload Data")
crop_file = st.sidebar.file_uploader("Upload Crop Data (.xlsx)", type=["xlsx"])
climate_files = st.sidebar.file_uploader("Upload Climate Data for Province (.xlsx)", type=["xlsx"], accept_multiple_files=True)

# --- Load Data ---
def load_crop_data(file):
    return pd.read_excel(file)

def load_climate_data(files):
    # files is a list of uploaded files
    df_list = []
    for f in files:
        temp_df = pd.read_excel(f)
        df_list.append(temp_df)
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

# --- Suitability Calculation ---
def check_failures(row, crop):
    failures = []
    if not (row['Rainfall Min'] >= crop['Rainfall Min']):
        failures.append('Rainfall Min')
    if not (row['Rainfall Max'] <= crop['Rainfall Max']):
        failures.append('Rainfall Max')
    if not (row['Temp Min'] >= crop['Temp Min']):
        failures.append('Temp Min')
    if not (row['Temp Max'] <= crop['Temp Max']):
        failures.append('Temp Max')
    if row['Drought Tolerance'] != crop['Drought Tolerance']:
        failures.append('Drought Tolerance')
    if row['Suitable Köppen Zones'] != crop['Suitable Köppen Zones']:
        failures.append('Suitable Köppen Zones')
    if row['Soil Texture'] != crop['Soil Texture']:
        failures.append('Soil Texture')
    if row['Drainage Preference'] != crop['Drainage Preference']:
        failures.append('Drainage Preference')
    if row['Irrigation Need'] != crop['Irrigation Need']:
        failures.append('Irrigation Need')
    return ', '.join(failures) if failures else 'None'

def calculate_suitability(climate_df, crop_df):
    results = []
    for _, crop in crop_df.iterrows():
        crop_name = crop['Crop Name']
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

            failures = check_failures(row, crop)

            results.append({
                'Crop Name': crop_name,
                'x': row['x'],
                'y': row['y'],
                'Suitability Score': score,
                'Failure Reasons': failures
            })
    return pd.DataFrame(results)

def categorize_score(score):
    if score >= 7:
        return 'High'
    elif score >= 4:
        return 'Moderate'
    elif score >= 0:
        return 'Low'

def get_area_shape(area):
    if area < 20:
        return "circle"
    elif area < 50:
        return "square"
    elif area < 100:
        return "triangle-up"
    elif area < 500:
        return "diamond"
    else:
        return "arrow-bar-up"

    # Identify mismatched and matched parameters for each row
def get_failure_reason(row, crop):
    reasons = []

    # Rainfall check
    if row["Rainfall Min"] < crop["Rainfall Min"] or row["Rainfall Max"] > crop["Rainfall Max"]:
        reasons.append("Rainfall")

    # Temperature check
    if row["Temp Min"] < crop["Temp Min"] or row["Temp Max"] > crop["Temp Max"]:
        reasons.append("Temperature")

    # Drought tolerance check
    if crop["Drought Tolerance"] != "Any" and row["Drought Tolerance"] != crop["Drought Tolerance"]:
        reasons.append("Drought Tolerance")

    # Suitable Köppen Zones check: parse strings to int lists
    try:
        crop_zones = [int(z.strip()) for z in str(crop["Suitable Köppen Zones"]).split(",")]
    except ValueError:
        crop_zones = []
    try:
        row_zones = [int(z.strip()) for z in str(row["Suitable Köppen Zones"]).split(",")]
    except ValueError:
        row_zones = []

    if not set(crop_zones).intersection(set(row_zones)):
        reasons.append("Köppen Zone")

    # Soil Texture check (assuming similar comma-separated string; adjust if needed)
    crop_soil = [s.strip() for s in str(crop["Soil Texture"]).split(",")]
    row_soil = [s.strip() for s in str(row["Soil Texture"]).split(",")]
    if not any(soil in row_soil for soil in crop_soil):
        reasons.append("Soil Texture")

    # Drainage Preference check (assuming similar comma-separated string)
    crop_drainage = [d.strip() for d in str(crop["Drainage Preference"]).split(",")]
    row_drainage = [d.strip() for d in str(row["Drainage Preference"]).split(",")]
    if not any(drain in row_drainage for drain in crop_drainage):
        reasons.append("Drainage")

    # Irrigation Need check
    if crop["Irrigation Need"] != "Any" and row["Irrigation Need"] != crop["Irrigation Need"]:
        reasons.append("Irrigation")

    return ", ".join(reasons) if reasons else "None"

# --- Main Logic ---
if crop_file and climate_files:
    with st.spinner("Processing data... Please wait."):
        crop_df = load_crop_data(crop_file)

        combined_climate_df = pd.DataFrame()
        dfs = []
        for f in climate_files:
            temp_df = pd.read_excel(f)
            temp_df['source_file'] = f.name
            dfs.append(temp_df)
        if dfs:
            combined_climate_df = pd.concat(dfs, ignore_index=True)

        if "Fallow land area" in combined_climate_df.columns:
            combined_climate_df.rename(columns={"Fallow land area": "area_ha"}, inplace=True)
            combined_climate_df['area_ha'] = pd.to_numeric(combined_climate_df['area_ha'], errors='coerce').fillna(0)
        else:
            combined_climate_df['area_ha'] = 0

        suitability_df = calculate_suitability(combined_climate_df, crop_df)
        suitability_df = suitability_df.merge(
            combined_climate_df[['x', 'y', 'area_ha', 'source_file']],
            on=['x', 'y'],
            how='left'
        )
        suitability_df['Suitability Category'] = suitability_df['Suitability Score'].apply(categorize_score)

    st.success("Data successfully processed.")

    st.sidebar.subheader("Filters")
    categories = suitability_df['Suitability Category'].unique().tolist()
    selected_categories = st.sidebar.multiselect("Suitability Category", options=categories, default=categories)

    all_failures = set()
    for fr in suitability_df['Failure Reasons']:
        if fr and fr != 'None':
            all_failures.update([f.strip() for f in fr.split(',')])
    selected_failures = st.sidebar.multiselect("Filter by Failure Reasons", options=sorted(list(all_failures)))

    selected_provinces = st.sidebar.multiselect(
        "Select Provinces",
        options=suitability_df['source_file'].unique(),
        default=suitability_df['source_file'].unique()
    )

    min_area = float(suitability_df['area_ha'].min(skipna=True))
    max_area = float(suitability_df['area_ha'].max(skipna=True))
    selected_area_range = st.sidebar.slider(
        "Select Fallow Land Area Range (ha)",
        min_value=min_area,
        max_value=max_area,
        value=(min_area, max_area),
        step=1.0
    )

    filtered_df = suitability_df[
        (suitability_df['Suitability Category'].isin(selected_categories)) &
        (suitability_df['source_file'].isin(selected_provinces)) &
        (suitability_df['area_ha'] >= selected_area_range[0]) &
        (suitability_df['area_ha'] <= selected_area_range[1])
    ]
    if selected_failures:
        pattern = '|'.join(selected_failures)
        filtered_df = filtered_df[filtered_df['Failure Reasons'].str.contains(pattern)]

    selected_crops = st.multiselect("Select Crops to Compare", crop_df['Crop Name'].unique(), default=crop_df['Crop Name'].unique()[0])
    filtered_df = filtered_df[filtered_df['Crop Name'].isin(selected_crops)]

    filtered_df["Shape"] = filtered_df["area_ha"].apply(get_area_shape)

    st.subheader("Suitability Map")
    color_map = {"High": "green", "Moderate": "orange", "Low": "red", "Unsuitable": "gray"}
    st.markdown(f"Showing areas between **{selected_area_range[0]} ha** and **{selected_area_range[1]} ha**")
    fig_map = px.scatter_mapbox(
        filtered_df,
        lat="y",
        lon="x",
        color="Suitability Category",
        color_discrete_map=color_map,
        symbol="Shape",
        hover_name="Crop Name",
        hover_data=["Suitability Score", "Failure Reasons", "area_ha", "source_file", "Shape"],
        mapbox_style="carto-positron",
        zoom=5,
        height=500
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Suitability Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=filtered_df, x="Suitability Score", hue="Crop Name", multiple="stack", bins=10, ax=ax)
    ax.set_xlabel("Suitability Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Summary Table")
    summary_rows = []
    for _, row in filtered_df.iterrows():
        crop_row = crop_df[crop_df["Crop Name"] == row["Crop Name"]].iloc[0]
        matching_climate_row = combined_climate_df[
            (combined_climate_df["x"] == row["x"]) &
            (combined_climate_df["y"] == row["y"]) &
            (combined_climate_df["source_file"] == row["source_file"])
        ]
        if not matching_climate_row.empty:
            climate_row = matching_climate_row.iloc[0]
            failure_reason = get_failure_reason(climate_row, crop_row)
            matched_params = 9 - failure_reason.count(",") if failure_reason != "None" else 9
        else:
            failure_reason = "Unavailable"
            matched_params = "N/A"

        summary_rows.append({
            "Crop Name": row["Crop Name"],
            "Grid Number on map": f"{row['x']}_{row['y']}",
            "Suitability Category": row["Suitability Category"],
            "Fallow Land Area (ha)": row["area_ha"],
            "Matched Parameters": matched_params,
            "Failed Parameters": failure_reason,
            "Failure Reason": failure_reason
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df)

    st.subheader("Download Results")
    def convert_df(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Suitability Results')
        return output.getvalue()

    st.download_button(
        label="Download Filtered Results as Excel",
        data=convert_df(summary_df),
        file_name="suitability_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Please upload both crop and climate datasets to begin.")

# --- Footer ---
st.markdown("---")
st.markdown("© Developed by Sasol Research & Technology: Feedstock (2025)")
