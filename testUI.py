import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
# import tensorflow as tf  
import plotly.express as px
import requests
import homepage
import os
import base64
import peatland
import chatbot
import subprocess
import footprint2
import nlpp
import credit2

VENV_APP1 = os.path.abspath("./genai")
APP1_PATH = os.path.abspath("chatbot.py")

background_image_path = "/Users/swastika/Downloads/peatland-8.jpg"

# st.title("AI-Powered Carbon Footprint Tracker 🌿")

page = st.sidebar.radio("Navigate", ["Home", "Peatland Analysis", "Knowledge Base", "Footprint Calculator", "Personalized Footprint Reduction", "Carbon Credits"])

if page == "Home":
    st.title("🌍 Climate Insights Dashboard")

    st.markdown("""
    Welcome to the Climate Insights Dashboard! Here, you can explore:
    - **Global Climate Trends**: CO₂ emissions, temperature changes, and extreme weather events.
    - **Carbon Sequestration**: Learn about natural carbon sinks like peatlands, forests, and oceans.
    - **Carbon Footprint Overview**: Track global & industry-specific emission trends.
    """)
    
    homepage.homepage()

elif page == "Peatland Analysis":
    peatland.peatland1()

elif page == "Knowledge Base":
    # python_exec = os.path.join(VENV_APP1, "bin", "python")
    # subprocess.Popen([python_exec, "-m", "streamlit", "run", APP1_PATH])
    chatbot.knowledge_base()

elif page == "Footprint Calculator":
    
    footprint2.footprint()

elif page == "Personalized Footprint Reduction":
    nlpp.recommend()

elif page == "Carbon Credits":
    st.header("AI-Powered Carbon Credit Recommender")
    st.write("Find verified carbon offset projects based on your preferences.")
    
    credit2.credit()

    # if st.button("Recommend Credits"):
    #     st.success("Suggested Carbon Credit: Reforestation Project in Indonesia (High Impact)")
