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

# --- Helper function ---
def is_multi_match(crop_val, land_val):
    """
    Returns True if any value in crop_val exists in land_val.
    Handles comma-separated strings, ignores spaces and order.
    """
    if pd.isna(crop_val) or pd.isna(land_val):
        return False
    try:
        crop_set = set(str(crop_val).replace(" ", "").split(","))
        land_set = set(str(land_val).replace(" ", "").split(","))
        return bool(crop_set & land_set)  # True if any overlap
    except Exception:
        return False

# --- Suitability Calculation ---
def check_failures(row, crop):
    failures = []
    if row['Rainfall Min'] < crop['Rainfall Min']:
        failures.append('Rainfall Min')
    if row['Rainfall Max'] > crop['Rainfall Max']:
        failures.append('Rainfall Max')
    if row['Temp Min'] < crop['Temp Min']:
        failures.append('Temp Min')
    if row['Temp Max'] > crop['Temp Max']:
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

@st.cache_data
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

# --- Main Logic ---
if crop_file and climate_files:
    with st.spinner("Processing data... Please wait."):
        crop_df = load_crop_data(crop_file)

        dfs = []
        for f in climate_files:
            temp_df = pd.read_excel(f)
            temp_df['source_file'] = f.name
            dfs.append(temp_df)
        combined_climate_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

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

    #st.sidebar.subheader("Filters")
    #categories = suitability_df['Suitability Category'].unique().tolist()
    #selected_categories = st.sidebar.multiselect("Suitability Category", options=categories, default=categories)

    all_failures = set()
    #if 'Failure Reasons' not in suitability_df.columns:
    #    suitability_df['Failure Reasons'] = 'None'

    #for fr in suitability_df['Failure Reasons']:
    #    if fr and fr != 'None':
    #        all_failures.update([f.strip() for f in fr.split(',')])
    #all_failures = sorted(list(all_failures))
    #selected_failures = st.sidebar.multiselect("Filter by Failure Reasons", options=all_failures)

    selected_provinces = st.sidebar.multiselect(
        "Select Provinces",
        options=suitability_df['source_file'].unique(),
        default=[]
    )

    selected_crops = st.multiselect("Select Crops to Compare", crop_df['Crop Name'].unique(), default=[])

    # --- Conditional Rendering ---
    if selected_provinces and selected_crops:
        filtered_df = suitability_df[
        (suitability_df['source_file'].isin(selected_provinces)) & 
        (suitability_df['Crop Name'].isin(selected_crops))]
        #if selected_failures:
         #   pattern = '|'.join(selected_failures)
         #   filtered_df = filtered_df[filtered_df['Failure Reasons'].str.contains(pattern)]
        
                # --- Provincial Breakdown Table ---
        st.subheader("Provincial Summary")

        def compute_provincial_summary(df, crop_df):
            # Extract province name from source_file (remove extension)
            df = df.copy()
            df['Province'] = df['source_file'].str.replace('.xlsx', '', regex=False)
            
            # Merge bioenergy info from crop_df
            crop_info = crop_df[['Crop Name', 'bioenergy category', 'average power density']]
            df = df.merge(crop_info, on='Crop Name', how='left')
    
            summary = []
            for (province, crop_name), group in df.groupby(['Province', 'Crop Name']):
                avg_score = group['Suitability Score'].mean()
                total = len(group)
                high = (group['Suitability Category'] == 'High').sum()
                moderate = (group['Suitability Category'] == 'Moderate').sum()
                low = (group['Suitability Category'] == 'Low').sum()
        
                # Proportions
                high_pct = (high / total) * 100 if total > 0 else 0
                moderate_pct = (moderate / total) * 100 if total > 0 else 0
                low_pct = (low / total) * 100 if total > 0 else 0
        
                # Most common limiting factor
                failure_series = group['Failure Reasons'].str.split(',').explode().str.strip()
                failure_series = failure_series[failure_series != 'None']
                main_limiting = failure_series.value_counts().idxmax() if not failure_series.empty else 'None'
        
                # Get bioenergy info (same for the crop)
                bio_category = group['bioenergy category'].iloc[0] if 'bioenergy category' in group.columns else 'N/A'
                avg_power = group['average power density'].iloc[0] if 'average power density' in group.columns else 'N/A'
        
                summary.append({
                    'Province': province,
                    'Crop Name': crop_name,
                    'Average Suitability Score': round(avg_score, 2),
                    'High (%)': round(high_pct, 1),
                    'Moderate (%)': round(moderate_pct, 1),
                    'Low (%)': round(low_pct, 1),
                    'Main Limiting Factor': main_limiting,
                    'Bioenergy Category': bio_category,
                    'Average Power Density (W/m³)': avg_power
                })
        
            return pd.DataFrame(summary)


        st.dataframe(
            provincial_summary_df.style.format({
                'Average Suitability Score': "{:.2f}",
                'High (%)': "{:.1f}",
                'Moderate (%)': "{:.1f}",
                'Low (%)': "{:.1f}"
            }),
            use_container_width=True
        )

        # Suitability Map
        st.subheader("Suitability Map")
        plot_df = filtered_df.copy()
        plot_df['Plot Score'] = plot_df['Suitability Score']
        plot_df.loc[plot_df['Suitability Category'] == 'Unsuitable', 'Plot Score'] = -1
        color_scale = [
           [0.0, "lightgrey"],  # -1 mapped to 0 fraction for grey
           [0.0001, "red"],     # start of actual suitability
           [0.5, "orange"],     # medium
           [1.0, "green"]       # high
        ]

# Normalize scores for color scale (0–9 normal range)
        fig_map = px.scatter_mapbox(
         plot_df,
         lat="y",
         lon="x",
         color="Plot Score",
         color_continuous_scale=color_scale,
         range_color=[-1, 9],  # include -1 for grey
         hover_name="Crop Name",
         hover_data=["Suitability Score", "Suitability Category", "Failure Reasons", "area_ha"],
         mapbox_style="open-street-map",
         zoom=7,
         height=500
     )
        st.plotly_chart(fig_map, use_container_width=True)

        # Let the user pick one crop for detailed analysis
        selected_crop = st.selectbox("Select Crop for Detailed Analysis", filtered_df['Crop Name'].unique())

        # Filter the plot_df for the selected crop
        plot_crop_df = plot_df[plot_df['Crop Name'] == selected_crop]

        # Pie Chart: Suitability Category Breakdown
        st.plotly_chart(
           px.pie(plot_crop_df, names='Suitability Category', title=f"Suitability Categories for {selected_crop}"),
           use_container_width=True)

        # Bar Chart: Top Performing Crops by Area
        top_area_df = suitability_df[suitability_df['Suitability Category'] == 'High']
        bar_df = top_area_df.groupby('Crop Name')['area_ha'].sum().reset_index().sort_values(by='area_ha', ascending=False).head(10)
        st.plotly_chart(
            px.bar(bar_df, x='Crop Name', y='area_ha', title= "Crop performace by area (ha) classifed as highly suitable"),
            use_container_width=True
        )

    else:
        st.warning("Please select at least one crop and one province to view the results.")
        st.stop()

# --- Footer ---
st.markdown("---")
st.markdown("© Developed by Sasol Research & Technology: Feedstock (2025)")


























