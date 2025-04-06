import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import sqlite3
import plotly.express as px
import plotly.graph_objects as go

PEATLANDS = {
    "Riau, Indonesia": (-0.5, 102.5),
    "Amazon, Brazil": (-3.0, -60.0),
    "Congo Basin, Africa": (1.5, 17.5),
    "Sundaland, Malaysia": (2.5, 102.0),
    "Hudson Bay Lowlands, Canada": (55.0, -85.0),
}

df = pd.DataFrame({
    'Date': [2018, 2021, 2022, 2023, 2024],
    'NDVI': [1, 1, 0.665254, 0.64197, 0.702914],
    'NDWI': [0.711013, 0.626983, 0.614526, 0.612952, 0.499578],
    'NDMI': [0.3099, 0.3027745, 0.295649, 0.285479, 0.269119]
})

df['Date'] = pd.to_datetime(df['Date'], format='%Y')

df.set_index('Date', inplace=True)

#def fit_arima_and_forecast(data, order=(1, 1, 1), steps=2):
def fit_arima_and_forecast(data, series_name, order=(1, 1, 1), steps=2):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)

    last_year = data.index[-1]
    forecast_years = [last_year + pd.DateOffset(years=i) for i in range(1, steps + 1)]
    
    forecast_df = pd.DataFrame({'Predicted': list(forecast)}, index=[year.year for year in forecast_years])
    forecast_df.index.name = 'Year'  
    
    return forecast_df


def insert_request(name, phone, email, latitude=None, longitude=None, id_proof_data=None, nir_data=None, swir_data=None, red_data=None, green_data=None):
    DB_PATH = "/Users/swastika/Carbon Footprint App/peatland_requests.db"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    data = {
        "name": name,
        "phone": phone,
        "email": email,
        "latitude": latitude,
        "longitude": longitude,
        "id_proof": id_proof_data,
        "nir_image": nir_data,
        "swir_image": swir_data,
        "red_image": red_data,
        "green_image": green_data
    }

    columns = [key for key in data if data[key] is not None]
    values = [data[key] for key in columns]

    query = f"INSERT INTO peatland_requests ({', '.join(columns)}) VALUES ({', '.join(['?' for _ in columns])})"

    try:
        cursor.execute(query, values)
        conn.commit()
        print("Data inserted successfully!")
        st.cache_data.clear()

    except sqlite3.Error as e:
        print("SQLite Error:", e)
    finally:
        conn.close()


tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series Analysis", "📊 Correlation Analysis", "🗺️ Spatial & Clustering Analysis", "📥 Upload & Analyze"])

# --- Time Series Analysis ---
with tab1:
    st.info("🔍 This analysis is based on remote sensing data from **Riau, Indonesia**.")
    st.markdown("## 📉 ARIMA Time Series Forecasting")

    col1, col2, col3 = st.columns([1, 1, 1])  

    ndvi_forecast = fit_arima_and_forecast(df['NDVI'], 'NDVI')  
    ndwi_forecast = fit_arima_and_forecast(df['NDWI'], 'NDWI')
    ndmi_forecast = fit_arima_and_forecast(df['NDMI'], 'NDMI') 

    with col1:
        st.metric(label="NDVI Forecast", value=round(ndvi_forecast.iloc[-1],4) if not ndvi_forecast.empty else "No Data")

    with col2:
        st.metric(label="NDWI Forecast", value=round(ndwi_forecast.iloc[-1],4) if not ndwi_forecast.empty else "No Data")

    with col3:
        st.metric(label="NDMI Forecast", value=round(ndmi_forecast.iloc[-1],4) if not ndmi_forecast.empty else "No Data")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.write("📌 **NDVI Forecast Values:**")
        ndvi_forecast_display = ndvi_forecast.copy()
        ndvi_forecast_display.index = ndvi_forecast_display.index.astype(str) 
        st.write(ndvi_forecast_display, use_container_width=True)
    with col2:
        st.write("📌 **NDWI Forecast Values:**")
        ndwi_forecast_display = ndwi_forecast.copy()
        ndwi_forecast_display.index = ndwi_forecast_display.index.astype(str) 
        st.write(ndwi_forecast_display, use_container_width=True)
    with col3:
        st.write("📌 **NDMI Forecast Values:**")
        ndmi_forecast_display = ndmi_forecast.copy()
        ndmi_forecast_display.index = ndmi_forecast_display.index.astype(str)  
        st.write(ndmi_forecast_display)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        fig_ndvi = px.line(df, x=df.index, y='NDVI', title='NDVI Over Time')
        fig_ndvi.update_layout(width=350, height=300)
        st.plotly_chart(fig_ndvi, use_container_width=True)
    with col2:
        fig_ndwi = px.line(df, x=df.index, y='NDWI', title='NDWI Over Time')
        fig_ndwi.update_layout(width=350, height=300)
        st.plotly_chart(fig_ndwi, use_container_width=True)
    with col3:
        fig_ndmi = px.line(df, x=df.index, y='NDMI', title='NDMI Over Time')
        fig_ndmi.update_layout(width=350, height=300)
        st.plotly_chart(fig_ndmi, use_container_width=True)

# --- Correlation Analysis ---
with tab2:
    st.info("🔍 This analysis is based on remote sensing data from **Riau, Indonesia**.")
    st.markdown("## 🔬 Correlation Analysis")

    correlation_matrix = df.corr()
    st.dataframe(correlation_matrix.style.background_gradient(cmap="coolwarm"))

# --- Spatial and Clustering Analysis ---
with tab3:
    st.info("🔍 This analysis is based on remote sensing data from **Riau, Indonesia**.")
    st.markdown("## 🌍 Spatial and Clustering Analysis")

    #st.markdown("##### 🛰️ Satellite-Derived Change Maps")
    st.markdown(
        "<h5 style='text-align: center;'>🛰️ Satellite-Derived Change Maps</h5>", 
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.image("/Users/swastika/IIRS/ndmi_diff_map.png", caption="NDMI Difference Map", use_container_width=True)
        
    with col2:
        st.image("/Users/swastika/IIRS/ndvi_diff_map.png", caption="NDVI Difference Map", use_container_width=True)

    with col3:
        st.image("/Users/swastika/IIRS/ndwi_diff_map.png", caption="NDWI Difference Map", use_container_width=True)

    st.markdown("<br><hr style='border:1px solid #ccc'><br>", unsafe_allow_html=True)

    # --- heatmap ---
    df_heatmap = df.copy()
    df_heatmap.index = df_heatmap.index.year  
    st.markdown("### 🔥 Time-Series Heatmaps for Peatland Changes")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### NDVI Heatmap")
        fig, ax = plt.subplots(figsize=(5, 3))  # Adjust figure size
        sns.heatmap(df_heatmap[['NDVI']].T, cmap="RdYlGn", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("### NDWI Heatmap")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(df_heatmap[['NDWI']].T, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

    with col3:
        st.markdown("### NDMI Heatmap")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(df_heatmap[['NDMI']].T, cmap="YlOrBr", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)


with tab4:
    st.markdown("## 📝 Submit Your Request for Peatland Analysis")
    st.markdown("💡 **Want the same analysis for your region?** Choose a peatland below or enter coordinates.")

    # --- Form ---
    with st.form("user_details_form"):
        st.markdown("### 👤 User Information")

        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("🆔 Full Name", placeholder="Enter your name")
            phone = st.text_input("📞 Phone Number", placeholder="Enter your phone")
        with col2:
            email = st.text_input("📧 Email Address", placeholder="Enter your email")
            id_proof = st.file_uploader("🆔 Upload ID Proof (PDF, JPG, PNG)", type=["pdf", "jpg", "png"])


        st.markdown("<br><hr style='border:1px solid #ccc'><br>", unsafe_allow_html=True)

        # st.markdown("### 🌍 Peatland Location")
        # col1, col2 = st.columns(2)
        # latitude = col1.number_input("📍 Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
        # longitude = col2.number_input("📍 Longitude", min_value=-180.0, max_value=180.0, format="%.6f")
        st.markdown("### 🌍 Select a Peatland or Enter Coordinates")
        selected_peatland = st.selectbox("Choose a peatland", ["Custom Location"] + list(PEATLANDS.keys()))
        latitude, longitude = None, None

        if selected_peatland == "Custom Location":
            col1, col2 = st.columns(2)
            latitude, longitude = col1.number_input("📍 Latitude", -90.0, 90.0), col2.number_input("📍 Longitude", -180.0, 180.0)
        else:
            latitude, longitude = PEATLANDS[selected_peatland]

        st.markdown("<br><hr style='border:1px solid #ccc'><br>", unsafe_allow_html=True)
        
        st.markdown("### 🛰️ Upload Satellite Images")
        col1, col2 = st.columns(2)
        with col1:
            nir_file = st.file_uploader("🌑 Upload NIR Band", type=["tif", "png", "jpg", "jp2"])
            swir_file = st.file_uploader("🔥 Upload SWIR Band", type=["tif", "png", "jpg", "jp2"])
        with col2:
            red_file = st.file_uploader("🌅 Upload RED Band", type=["tif", "png", "jpg", "jp2"])
            green_file = st.file_uploader("🌿 Upload GREEN Band", type=["tif", "png", "jpg", "jp2"])

        submit_button = st.form_submit_button("🔍 Submit & Analyze")
    
    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False

    if submit_button:
        if not name or not phone or not email:
            st.error("⚠️ Please fill in Name, Phone, and Email!")
        else:
            def convert_to_binary(file):
                return file.read() if file else None

            id_proof_data = convert_to_binary(id_proof) if id_proof else None
            nir_data = convert_to_binary(nir_file) if nir_file else None
            swir_data = convert_to_binary(swir_file) if swir_file else None
            red_data = convert_to_binary(red_file) if red_file else None
            green_data = convert_to_binary(green_file) if green_file else None

            if (nir_data or swir_data or red_data or green_data) or (latitude and longitude):
                has_images = any([nir_data, swir_data, red_data, green_data])
                if has_images:
                    latitude, longitude = None, None
                insert_request(name, phone, email, latitude, longitude, id_proof_data, nir_data, swir_data, red_data, green_data)
                st.session_state["submitted"] = True 
                st.success("Request submitted successfully!")
            else:
                st.error("⚠️ Please provide either satellite images OR latitude & longitude!")

############################

def peatland1():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA
    import sqlite3
    import plotly.express as px
    import plotly.graph_objects as go

    PEATLANDS = {
        "Riau, Indonesia": (-0.5, 102.5),
        "Amazon, Brazil": (-3.0, -60.0),
        "Congo Basin, Africa": (1.5, 17.5),
        "Sundaland, Malaysia": (2.5, 102.0),
        "Hudson Bay Lowlands, Canada": (55.0, -85.0),
    }

    df = pd.DataFrame({
        'Date': [2018, 2021, 2022, 2023, 2024],
        'NDVI': [1, 1, 0.665254, 0.64197, 0.702914],
        'NDWI': [0.711013, 0.626983, 0.614526, 0.612952, 0.499578],
        'NDMI': [0.3099, 0.3027745, 0.295649, 0.285479, 0.269119]
    })

    df['Date'] = pd.to_datetime(df['Date'], format='%Y')

    df.set_index('Date', inplace=True)

    #def fit_arima_and_forecast(data, order=(1, 1, 1), steps=2):
    def fit_arima_and_forecast(data, series_name, order=(1, 1, 1), steps=2):
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)

        last_year = data.index[-1]
        forecast_years = [last_year + pd.DateOffset(years=i) for i in range(1, steps + 1)]
        
        forecast_df = pd.DataFrame({'Predicted': list(forecast)}, index=[year.year for year in forecast_years])
        forecast_df.index.name = 'Year'  
        
        return forecast_df


    def insert_request(name, phone, email, latitude=None, longitude=None, id_proof_data=None, nir_data=None, swir_data=None, red_data=None, green_data=None):
        DB_PATH = "/Users/swastika/Carbon Footprint App/peatland_requests.db"
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        data = {
            "name": name,
            "phone": phone,
            "email": email,
            "latitude": latitude,
            "longitude": longitude,
            "id_proof": id_proof_data,
            "nir_image": nir_data,
            "swir_image": swir_data,
            "red_image": red_data,
            "green_image": green_data
        }

        columns = [key for key in data if data[key] is not None]
        values = [data[key] for key in columns]

        query = f"INSERT INTO peatland_requests ({', '.join(columns)}) VALUES ({', '.join(['?' for _ in columns])})"

        try:
            cursor.execute(query, values)
            conn.commit()
            print("Data inserted successfully!")
            st.cache_data.clear()

        except sqlite3.Error as e:
            print("SQLite Error:", e)
        finally:
            conn.close()


    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series Analysis", "📊 Correlation Analysis", "🗺️ Spatial & Clustering Analysis", "📥 Upload & Analyze"])

    # --- Time Series Analysis ---
    with tab1:
        st.info("🔍 This analysis is based on remote sensing data from **Riau, Indonesia**.")
        st.markdown("## 📉 ARIMA Time Series Forecasting")

        col1, col2, col3 = st.columns([1, 1, 1])  

        ndvi_forecast = fit_arima_and_forecast(df['NDVI'], 'NDVI')  
        ndwi_forecast = fit_arima_and_forecast(df['NDWI'], 'NDWI')
        ndmi_forecast = fit_arima_and_forecast(df['NDMI'], 'NDMI') 

        with col1:
            st.metric(label="NDVI Forecast", value=round(ndvi_forecast.iloc[-1],4) if not ndvi_forecast.empty else "No Data")

        with col2:
            st.metric(label="NDWI Forecast", value=round(ndwi_forecast.iloc[-1],4) if not ndwi_forecast.empty else "No Data")

        with col3:
            st.metric(label="NDMI Forecast", value=round(ndmi_forecast.iloc[-1],4) if not ndmi_forecast.empty else "No Data")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.write("📌 **NDVI Forecast Values:**")
            ndvi_forecast_display = ndvi_forecast.copy()
            ndvi_forecast_display.index = ndvi_forecast_display.index.astype(str) 
            st.write(ndvi_forecast_display, use_container_width=True)
        with col2:
            st.write("📌 **NDWI Forecast Values:**")
            ndwi_forecast_display = ndwi_forecast.copy()
            ndwi_forecast_display.index = ndwi_forecast_display.index.astype(str) 
            st.write(ndwi_forecast_display, use_container_width=True)
        with col3:
            st.write("📌 **NDMI Forecast Values:**")
            ndmi_forecast_display = ndmi_forecast.copy()
            ndmi_forecast_display.index = ndmi_forecast_display.index.astype(str)  
            st.write(ndmi_forecast_display)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            fig_ndvi = px.line(df, x=df.index, y='NDVI', title='NDVI Over Time')
            fig_ndvi.update_layout(width=350, height=300)
            st.plotly_chart(fig_ndvi, use_container_width=True)
        with col2:
            fig_ndwi = px.line(df, x=df.index, y='NDWI', title='NDWI Over Time')
            fig_ndwi.update_layout(width=350, height=300)
            st.plotly_chart(fig_ndwi, use_container_width=True)
        with col3:
            fig_ndmi = px.line(df, x=df.index, y='NDMI', title='NDMI Over Time')
            fig_ndmi.update_layout(width=350, height=300)
            st.plotly_chart(fig_ndmi, use_container_width=True)

    # --- Correlation Analysis ---
    with tab2:
        st.info("🔍 This analysis is based on remote sensing data from **Riau, Indonesia**.")
        st.markdown("## 🔬 Correlation Analysis")

        correlation_matrix = df.corr()
        st.dataframe(correlation_matrix.style.background_gradient(cmap="coolwarm"))

    # --- Spatial and Clustering Analysis ---
    with tab3:
        st.info("🔍 This analysis is based on remote sensing data from **Riau, Indonesia**.")
        st.markdown("## 🌍 Spatial and Clustering Analysis")

        #st.markdown("##### 🛰️ Satellite-Derived Change Maps")
        st.markdown(
            "<h5 style='text-align: center;'>🛰️ Satellite-Derived Change Maps</h5>", 
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.image("/Users/swastika/IIRS/ndmi_diff_map.png", caption="NDMI Difference Map", use_container_width=True)
            
        with col2:
            st.image("/Users/swastika/IIRS/ndvi_diff_map.png", caption="NDVI Difference Map", use_container_width=True)

        with col3:
            st.image("/Users/swastika/IIRS/ndwi_diff_map.png", caption="NDWI Difference Map", use_container_width=True)

        st.markdown("<br><hr style='border:1px solid #ccc'><br>", unsafe_allow_html=True)

        # --- heatmap ---
        df_heatmap = df.copy()
        df_heatmap.index = df_heatmap.index.year  
        st.markdown("### 🔥 Time-Series Heatmaps for Peatland Changes")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("### NDVI Heatmap")
            fig, ax = plt.subplots(figsize=(5, 3))  # Adjust figure size
            sns.heatmap(df_heatmap[['NDVI']].T, cmap="RdYlGn", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)

        with col2:
            st.markdown("### NDWI Heatmap")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(df_heatmap[['NDWI']].T, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)

        with col3:
            st.markdown("### NDMI Heatmap")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(df_heatmap[['NDMI']].T, cmap="YlOrBr", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)


    with tab4:
        st.markdown("### 📝 Submit Your Request for Peatland Analysis")
        st.markdown("💡 **Want the same analysis for your region?** Choose a peatland below or enter coordinates.")

        # --- Form ---
        with st.form("user_details_form"):
            st.markdown("### 👤 User Information")

            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("🆔 Full Name", placeholder="Enter your name")
                phone = st.text_input("📞 Phone Number", placeholder="Enter your phone")
            with col2:
                email = st.text_input("📧 Email Address", placeholder="Enter your email")
                id_proof = st.file_uploader("🆔 Upload ID Proof (PDF, JPG, PNG)", type=["pdf", "jpg", "png"])


            st.markdown("<br><hr style='border:1px solid #ccc'><br>", unsafe_allow_html=True)

            # st.markdown("### 🌍 Peatland Location")
            # col1, col2 = st.columns(2)
            # latitude = col1.number_input("📍 Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
            # longitude = col2.number_input("📍 Longitude", min_value=-180.0, max_value=180.0, format="%.6f")
            st.markdown("### 🌍 Select a Peatland or Enter Coordinates")
            selected_peatland = st.selectbox("Choose a peatland", ["Custom Location"] + list(PEATLANDS.keys()))
            latitude, longitude = None, None

            if selected_peatland == "Custom Location":
                col1, col2 = st.columns(2)
                latitude, longitude = col1.number_input("📍 Latitude", -90.0, 90.0), col2.number_input("📍 Longitude", -180.0, 180.0)
            else:
                latitude, longitude = PEATLANDS[selected_peatland]

            st.markdown("<br><hr style='border:1px solid #ccc'><br>", unsafe_allow_html=True)
            
            st.markdown("### 🛰️ Upload Satellite Images")
            col1, col2 = st.columns(2)
            with col1:
                nir_file = st.file_uploader("🌑 Upload NIR Band", type=["tif", "png", "jpg", "jp2"])
                swir_file = st.file_uploader("🔥 Upload SWIR Band", type=["tif", "png", "jpg", "jp2"])
            with col2:
                red_file = st.file_uploader("🌅 Upload RED Band", type=["tif", "png", "jpg", "jp2"])
                green_file = st.file_uploader("🌿 Upload GREEN Band", type=["tif", "png", "jpg", "jp2"])

            submit_button = st.form_submit_button("🔍 Submit & Analyze")
        
        if "submitted" not in st.session_state:
            st.session_state["submitted"] = False

        if submit_button:
            if not name or not phone or not email:
                st.error("⚠️ Please fill in Name, Phone, and Email!")
            else:
                def convert_to_binary(file):
                    return file.read() if file else None

                id_proof_data = convert_to_binary(id_proof) if id_proof else None
                nir_data = convert_to_binary(nir_file) if nir_file else None
                swir_data = convert_to_binary(swir_file) if swir_file else None
                red_data = convert_to_binary(red_file) if red_file else None
                green_data = convert_to_binary(green_file) if green_file else None

                if (nir_data or swir_data or red_data or green_data) or (latitude and longitude):
                    has_images = any([nir_data, swir_data, red_data, green_data])
                    if has_images:
                        latitude, longitude = None, None
                    insert_request(name, phone, email, latitude, longitude, id_proof_data, nir_data, swir_data, red_data, green_data)
                    st.session_state["submitted"] = True 
                    st.success("Request submitted successfully!")
                else:
                    st.error("⚠️ Please provide either satellite images OR latitude & longitude!")

if __name__ == "__main__":
    peatland1()