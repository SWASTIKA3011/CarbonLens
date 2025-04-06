import streamlit as st
import plotly.express as px
import pandas as pd
import requests

#st.set_page_config(page_title='Climate Insights', layout='wide')

#st.title("🌍 Climate Insights Dashboard")

# st.markdown("""
# Welcome to the Climate Insights Dashboard! Here, you can explore:
# - **Global Climate Trends**: CO₂ emissions, temperature changes, and extreme weather events.
# - **Carbon Sequestration**: Learn about natural carbon sinks like peatlands, forests, and oceans.
# - **Carbon Footprint Overview**: Track global & industry-specific emission trends.
# """)

def homepage():
    st.subheader("🌱 Natural Carbon Sequestration")
    st.markdown("""
    - **Peatlands**: Store 30% of global soil carbon despite covering only 3% of land.
    - **Forests**: Act as a major carbon sink, absorbing ~7.6 billion metric tons of CO₂ annually.
    - **Oceans**: Absorb ~25% of human-made CO₂ emissions each year.
    """)

    st.subheader("📊 Carbon Footprint Trends")
    st.markdown("""
    - **Industry Contributions**: Energy (73%), Agriculture (18%), Transport (16%).
    - **Per Capita Emissions**: USA (15.5 tons), India (1.9 tons), China (7.4 tons).
    """)

    st.write("🔍 Data sources: Global Warming API, Research Reports")