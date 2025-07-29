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
def load_crop_data(file):
    return pd.read_excel(file)

def load_climate_data(files):
    df_list = []
    for f in files:
        temp_df = pd.read_excel(f)
        df_list.append(temp_df)
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def is_multi_match(crop_val, land_val):
    # Handle None or NaN as no match
    if pd.isna(crop_val) or pd.isna(land_val):
        return False
    try:
        crop_set = set(str(crop_val).split(","))
        land_set = set(str(land_val).split(","))
        # Strip spaces and compare sets
        crop_set = set(x.strip() for x in crop_set)
        land_set = set(x.strip() for x in land_set)
        return len(crop_set.intersection(land_set)) > 0
    except Exception:
        return False

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
    if str(row['Drought Tolerance']).strip() != str(crop['Drought Tolerance']).strip():
        failures.append('Drought Tolerance')
    if not is_multi_match(crop['Suitable Köppen Zones'], row['Suitable Köppen Zones']):
        failures.append('Suitable Köppen Zones')
    if not is_multi_match(crop['Soil Texture'], row['Soil Texture']):
        failures.append('Soil Texture')
    if not is_multi_match(crop['Drainage Preference'], row['Drainage Preference']):
        failures.append('Drainage Preference')
    if str(row['Irrigation Need']).strip().lower() != str(crop['Irrigation Need']).strip().lower():
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
                str(row['Drought Tolerance']).strip() == str(crop['Drought Tolerance']).strip(),
                is_multi_match(crop['Suitable Köppen Zones'], row['Suitable Köppen Zones']),
                is_multi_match(crop['Soil Texture'], row['Soil Texture']),
                is_multi_match(crop['Drainage Preference'], row['Drainage Preference']),
                str(row['Irrigation Need']).strip().lower() == str(crop['Irrigation Need']).strip().lower()
            ])

            results.append({
                'Crop Name': crop_name,
                'x': row['x'],
                'y': row['y'],
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

# Identify mismatched and matched parameters for each row
def get_failure_reason(row, crop):
    reasons = []

    if row["Rainfall Min"] < crop["Rainfall Min"] or row["Rainfall Max"] > crop["Rainfall Max"]:
        reasons.append("Rainfall")
    if row["Temp Min"] < crop["Temp Min"] or row["Temp Max"] > crop["Temp Max"]:
        reasons.append("Temperature")
    if crop["Drought Tolerance"] != "Any" and row["Drought Tolerance"] != crop["Drought Tolerance"]:
        reasons.append("Drought Tolerance")

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

    crop_soil = [s.strip() for s in str(crop["Soil Texture"]).split(",")]
    row_soil = [s.strip() for s in str(row["Soil Texture"]).split(",")]
    if not any(soil in row_soil for soil in crop_soil):
        reasons.append("Soil Texture")

    crop_drainage = [d.strip() for d in str(crop["Drainage Preference"]).split(",")]
    row_drainage = [d.strip() for d in str(row["Drainage Preference"]).split(",")]
    if not any(drain in row_drainage for drain in crop_drainage):
        reasons.append("Drainage")

    if crop["Irrigation Need"] != "Any" and row["Irrigation Need"] != crop["Irrigation Need"]:
        reasons.append("Irrigation")

    return ", ".join(reasons) if reasons else "None"

# --- Main Logic ---
if crop_file and climate_files:
    with st.spinner("Processing data... Please wait."):
        crop_df = load_crop_data(crop_file)

        dfs = []
        for f in climate_files:
            temp_df = pd.read_excel(f)
            temp_df['source_file'] = f.name
            dfs.append(temp_df)
        if dfs:
            combined_climate_df = pd.concat(dfs, ignore_index=True)
        else:
            combined_climate_df = pd.DataFrame()

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
    if 'Failure Reasons' not in suitability_df.columns:
        suitability_df['Failure Reasons'] = 'None'

    for fr in suitability_df['Failure Reasons']:
        if fr and fr != 'None':
            all_failures.update([f.strip() for f in fr.split(',')])
    all_failures = sorted(list(all_failures))
    selected_failures = st.sidebar.multiselect("Filter by Failure Reasons", options=all_failures)

    selected_provinces = st.sidebar.multiselect(
        "Select Provinces",
        options=suitability_df['source_file'].unique(),
        default=suitability_df['source_file'].unique()
    )

    filtered_df = suitability_df[
        (suitability_df['Suitability Category'].isin(selected_categories)) &
        (suitability_df['source_file'].isin(selected_provinces))
    ]
    if selected_failures:
        pattern = '|'.join(selected_failures)
        filtered_df = filtered_df[filtered_df['Failure Reasons'].str.contains(pattern)]

    selected_crops = st.multiselect("Select Crops to Compare", crop_df['Crop Name'].unique(), default=crop_df['Crop Name'].unique()[0])
    filtered_df = filtered_df[filtered_df['Crop Name'].isin(selected_crops)]

    # Suitability Map
    st.subheader("Suitability Map")
    color_map = {
        "High": "green",
        "Moderate": "orange",
        "Low": "red",
        "Unsuitable": "gray"
    }
    fig_map = px.scatter_mapbox(
        filtered_df,
        lat="y",
        lon="x",
        color="Suitability Category",
        color_discrete_map=color_map,
        hover_name="Crop Name",
        hover_data=["Suitability Score", "Failure Reasons", "area_ha", "source_file"],
        mapbox_style="open-street-map",
        zoom=7,
        height=500
     )
    st.plotly_chart(fig_map, use_container_width=True)

    # Histogram
    st.subheader("Suitability Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=filtered_df, x="Suitability Score", hue="Crop Name", multiple="stack", bins=10, ax=ax)
    ax.set_xlabel("Suitability Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Interactive Analysis")

    selected_crop = st.selectbox("Select Crop", suitability_df['Crop Name'].unique())
    filtered_df = suitability_df[suitability_df['Crop Name'] == selected_crop]

    # Pie Chart: Suitability Category Breakdown
    st.plotly_chart(
        px.pie(filtered_df, names='Suitability Category', title=f"Suitability Categories for {selected_crop}"),
        use_container_width=True
    )

    # Bar Chart: Top Performing Crops by Area
    top_area_df = suitability_df[suitability_df['Suitability Category'] == 'High']
    bar_df = top_area_df.groupby('Crop Name')['area_ha'].sum().reset_index().sort_values(by='area_ha', ascending=False).head(10)
    st.plotly_chart(
        px.bar(bar_df, x='Crop Name', y='area_ha', title='Top Crops by High Suitability Area (ha)'),
        use_container_width=True
    )

    # Pie Chart: Failure Reason Breakdown
    failure_series = suitability_df['Failure Reasons'].str.split(', ').explode()
    failure_counts = failure_series.value_counts().reset_index()
    failure_counts.columns = ['Failure Reason', 'Count']
    st.plotly_chart(
        px.pie(failure_counts, names='Failure Reason', values='Count', title='Failure Reasons Breakdown'),
        use_container_width=True
    )

    # Download Button
    def convert_df(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Suitability')
        return output.getvalue()
        
    # --- Download Button ---
    st.subheader("Download Results")

    st.download_button(
        label="Download Full Suitability Data",
        data=convert_df(suitability_df),
        file_name="crop_suitability_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    suitability_colors = {
    "High": "green",
    "Moderate": "orange",
    "Low": "red",
    "Unsuitable": "gray"
     }
    # --- Fallow Land Area Threshold Maps ---
    if selected_crops:
       selected_crop = selected_crops[0]
    else:
       selected_crop = "Selected Crop"  # fallback

    st.subheader(f"Maps by Fallow Land Area Threshold for {selected_crop}")
    area_thresholds = [20, 50, 100, 500]
    
    for threshold in area_thresholds:
        st.markdown(f"**Fallow Land Area ≥ {threshold} Ha**")
        combined_climate_df.rename(columns={"Fallow land area": "area_ha"}, inplace=True)


        if threshold_df.empty:
            st.warning(f"No coordinates meet the threshold of {threshold} Ha.")
        else:
            fig_thresh = px.scatter_mapbox(
                threshold_df,
                lat="y",
                lon="x",
                color="Suitability Category",
                color_discrete_map=suitability_colors,
                hover_name="Grid Number on map",
                zoom=5,
                height=500,
            )
            fig_thresh.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_thresh, use_container_width=True)

            # --- Download Button ---
            csv_bytes = threshold_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download CSV for Area ≥ {threshold} Ha",
                data=csv_bytes,
                file_name=f"{selected_crop}_fallow_{threshold}Ha.csv",
                mime="text/csv"
            )


# --- Grid-Level Summary Button ---
st.subheader("Grid-Level Suitability Summary")
if 'suitability_df' in locals():
    if st.button("Generate Grid-Level Summary"):
        with st.spinner("Processing grid-level summary..."):
            # Ensure categorization exists at point level
            suitability_df['Suitability Category'] = suitability_df['Suitability Score'].apply(categorize_score)

            # Create a grid identifier if not present
            suitability_df['Grid Number on map'] = suitability_df['x'].astype(str) + '_' + suitability_df['y'].astype(str)

            # Group by Grid Number and Crop Name
            grid_summary = suitability_df.groupby(['Grid Number on map', 'Crop Name']).agg({
                'Suitability Score': 'mean',
                'x': 'count'
            }).reset_index()

            grid_summary.rename(columns={
                'x': 'Number of Points',
                'Suitability Score': 'Average Score'
            }, inplace=True)

            # Categorize grid-level average scores
            grid_summary['Grid Suitability Category'] = grid_summary['Average Score'].apply(categorize_score)

            # Show top 10
            st.dataframe(grid_summary.head(10))

            # Download option
            grid_csv = grid_summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Grid-Level Summary as CSV",
                data=grid_csv,
                file_name="grid_level_suitability_summary.csv",
                mime='text/csv'
            )
else:
    st.info("Please upload and process data first to enable grid-level summary.")


# --- Footer ---
st.markdown("---")
st.markdown("© Developed by Sasol Research & Technology: Feedstock (2025)")
