import streamlit as st
import pandas as pd
import spacy
from fuzzywuzzy import process
import os
import google.generativeai as genai 
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Load NLP model
nlp = spacy.load("en_core_web_sm")

########### FOOD ############
food_df = pd.read_csv("/Users/Swastika/Downloads/35605.csv")
food_df = food_df[[
    "Entity", 
    "GHG emissions per kilogram (Poore & Nemecek, 2018)", 
    "Land use per kilogram (Poore & Nemecek, 2018)", 
    "Freshwater withdrawals per kilogram (Poore & Nemecek, 2018)",
    "Eutrophying emissions per kilogram (Poore & Nemecek, 2018)"
]]
food_df["Entity"] = food_df["Entity"].str.lower().str.strip()

########### CAR ############
fuel_df = pd.read_excel("/Users/Swastika/Downloads/ghg-conversion-factors-2024-FlatFormat_v1_1.xlsx", sheet_name='Sheet1')
fuel_df = fuel_df[[
    "Level 3",  
    "Column Text",  
    "GHG Conversion Factor 2024",  
    "UOM"  
]]
fuel_df['Column Text'] = fuel_df['Column Text'].apply(lambda x: str(x).lower() if not pd.isna(x) else 'unknown')

car_categories = ["Mini", "Supermini", "Lower medium", "Upper medium", "Executive", "Luxury", "Sports", "Dual purpose 4X4", "MPV"]

########### FLIGHT ############
flight_df = pd.read_excel("/Users/Swastika/Downloads/ghg-conversion-factors-2024-FlatFormat_v1_1.xlsx", sheet_name='Sheet2')
flight_df = flight_df[[
    "LCA Activity",  
    "Emission Factor (kgCO₂e/passenger-km)",  
    "Description"
]]
flight_df['LCA Activity'] = flight_df['LCA Activity'].astype(str).str.lower()
flight_keywords = {"flight", "airplane", "plane", "air travel", "airline"}

########### BIKE ############
df = pd.read_excel("/Users/Swastika/Downloads/ghg-conversion-factors-2024-FlatFormat_v1_1.xlsx", sheet_name='Sheet1')  
df.columns = df.columns.str.strip()
df = df[[
    "Level 2",  
    "Level 3",  
    "GHG Conversion Factor 2024",  
]]
df["Level 2"] = df["Level 2"].astype(str).str.lower().str.strip()
df["Level 3"] = df["Level 3"].astype(str).str.lower().str.strip()
bike_keywords = {"bike", "motorbike", "motorcycle", "scooter"}

########### WASTE ############
waste_df = pd.read_excel("/Users/Swastika/Downloads/ghg-emission-factors-hub-2025.xlsx", sheet_name='Sheet1')
waste_df.columns = waste_df.columns.str.strip()
waste_df["Material"] = waste_df["Material"].astype(str).str.lower().str.strip()
waste_materials = set(waste_df["Material"].tolist())


# Food Carbon Footprint Report
def generate_food_footprint_report(food_items_list):
    report_data = []
    for food in food_items_list:
        match = food_df[food_df["Entity"] == food]
        if match.empty:
            best_match = process.extractOne(food, food_df["Entity"].tolist())
            if best_match and best_match[1] > 60:
                match = food_df[food_df["Entity"] == best_match[0]]
        if not match.empty:
            report_data.append(match.iloc[0].to_dict())
    return pd.DataFrame(report_data)

# Vehicle Carbon Footprint Report
def generate_vehicle_report(user_input):
    detected_categories = extract_vehicle_data(user_input)
    
    if not detected_categories:
        detected_categories = ["Mini", "Supermini"]
    
    report_data = []
    for category in detected_categories:
        category_matches = fuel_df[fuel_df['Level 3'].str.contains(category, case=False, na=False)]
        for _, row in category_matches.iterrows():
            report_row = {
                'Vehicle Type': row['Level 3'],
                'Fuel Type': row['Column Text'],
                'GHG Emissions per km (kg CO2e)': row['GHG Conversion Factor 2024'],
            }
            report_data.append(report_row)
    
    report_df = pd.DataFrame(report_data)
    report_df.drop_duplicates(inplace=True)
    return report_df

# Flight Emissions Report
def generate_flight_report(user_input):
    if flight_found:
        print("\n📊 Flight Emissions Report:\n")
        return flight_df  
    else:
        print("\nNo flight-related terms found in input.")
        return pd.DataFrame()  

# Bike Emissions Report
def generate_bike_report(user_input):
    """Generate emissions report if bike-related terms are detected."""
    if bike_found:
        bike_df = df[df["Level 2"].str.contains("motorbike", case=False, na=False)]
        report_df = bike_df.drop_duplicates()
        report_df.drop(columns={'Level 2'}, inplace=True)
        report_df.rename(columns={'GHG Conversion Factor 2024':'GHG Conversion Factor per km', 'Level 3':'Motorbike Size'}, inplace=True)
        if not report_df.empty:
            print("\n📊 Motorbike Emissions Report:\n")
        else:
            print("\n⚠️ No motorbike data found in the emissions table.")
        
        return report_df
    else:
        print("\n⚠️ No motorbike-related words found in input.")
        return pd.DataFrame()  

# Waste Handling Report
def generate_waste_report(user_input):
    extracted_waste_info = extract_waste_data(user_input)
    report_data = []
    for waste in extracted_waste_info:
        match = waste_df[waste_df["Material"].str.contains(waste, na=False, case=False)]
        
        if match.empty:
            best_match, score = process.extractOne(waste, waste_materials)
            if score > 60:  
                match = waste_df[waste_df["Material"] == best_match]

        if not match.empty:
            for _, row in match.iterrows():
                report_row = {
                    'Material': row['Material'].capitalize(),
                    'Recycled (kgCO₂e/short ton)': row.get('Recycled', 'N/A'),
                    'Landfilled (kgCO₂e/short ton)': row.get('Landfilled', 'N/A'),
                    'Combusted (kgCO₂e/short ton)': row.get('Combusted', 'N/A'),
                    'Composted (kgCO₂e/short ton)': row.get('Composted', 'N/A'),
                    'Anaerobically Digested (Dry Digestate with Curing) (kgCO₂e/short ton)': row.get('Anaerobically Digested (Dry Digestate with Curing)', 'N/A'),
                    'Anaerobically Digested (Wet Digestate with Curing) (kgCO₂e/short ton)': row.get('Anaerobically Digested (Wet Digestate with Curing)', 'N/A')
                }
                report_data.append(report_row)

    report_df = pd.DataFrame(report_data)
    report_df.drop_duplicates(inplace=True)

    if report_df.empty:
        print("\n⚠️ No matching waste materials found in the database.")
        return pd.DataFrame()
    
    return report_df


# Streamlit UI
st.title("Carbon Footprint Tracker")
user_input = st.text_area("Enter your daily activities:")

if st.button("Analyze Carbon Footprint"):
    doc = nlp(user_input.lower())
    
    food_items = [token.text for token in doc if token.text in food_df["Entity"].tolist()]

    def extract_vehicle_data(text):
        doc = nlp(text)
        vehicle_items = set()
        detected_categories = set()

        for sent in doc.sents:
            for phrase in sent.noun_chunks:
                phrase_text = phrase.text.lower()
                if "car" in phrase_text:
                    vehicle_items.add(phrase_text)
                    for category in car_categories:
                        if category.lower() in phrase_text:
                            detected_categories.add(category)

        return list(detected_categories)
    

    flight_found = any(token.text in flight_keywords for token in doc)

    bike_found = any(token.text in bike_keywords for token in doc)


    def extract_waste_data(text):
        doc = nlp(text.lower())
        detected_waste_items = set()
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower().strip()
            if phrase in waste_materials:
                detected_waste_items.add(phrase)
        
        # Extract single words using lemmatization
        for token in doc:
            word = token.lemma_.lower().strip()
            if word in waste_materials:
                detected_waste_items.add(word)

        remaining_words = [token.lemma_.lower().strip() for token in doc if token.pos_ in ('NOUN', 'ADJ') and token.lemma_.lower().strip() not in detected_waste_items]
        for word in remaining_words:
            best_match, score = process.extractOne(word, waste_materials)
            if score > 80:  # Raise the threshold significantly!
                detected_waste_items.add(best_match)

        detected_waste_items = [item for item in detected_waste_items]

        return list(detected_waste_items)
    

    # Food Carbon Footprint Report
    with st.expander("🍽️ Food Carbon Footprint"):
        food_report = generate_food_footprint_report(food_items)
        st.dataframe(food_report)

    # Vehicle Emissions Report
    with st.expander("🚗 Vehicle Emissions"):
        vehicle_report = generate_vehicle_report(user_input)
        st.dataframe(vehicle_report)

    # Bike Emissions Report (Only if bike-related terms found)
    if bike_found:
        with st.expander("🏍️ Bike Emissions"):
            bike_report = generate_bike_report(user_input)
            st.dataframe(bike_report)

    # Flight Emissions Report (Only if flight-related terms found)
    if flight_found:
        with st.expander("✈️ Flight Emissions"):
            flight_report = generate_flight_report(user_input)
            st.dataframe(flight_report)

    # Waste Handling Report
    with st.expander("🗑️ Waste Handling Report"):
        waste_report = generate_waste_report(user_input)
        st.dataframe(waste_report)


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
def get_gemini_recommendation(user_text):
    model = genai.GenerativeModel("gemini-1.5-flash")  # Use Gemini-Pro model
    response = model.generate_content(f"Provide personalized eco-friendly recommendations based on this user's input:\n{user_text}")
    return response.text if response else "Sorry, I couldn't generate recommendations."

st.subheader("🔍 Personalized Sustainability Recommendations")
    
recommendation = get_gemini_recommendation(user_input)
    
st.write(recommendation)  # Display recommendations



#############################################################
def recommend():

    import streamlit as st
    import pandas as pd
    import spacy
    from fuzzywuzzy import process
    import os
    import google.generativeai as genai 
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    # Load NLP model
    nlp = spacy.load("en_core_web_sm")

    ########### FOOD ############
    food_df = pd.read_csv("/Users/Swastika/Downloads/35605.csv")
    food_df = food_df[[
        "Entity", 
        "GHG emissions per kilogram (Poore & Nemecek, 2018)", 
        "Land use per kilogram (Poore & Nemecek, 2018)", 
        "Freshwater withdrawals per kilogram (Poore & Nemecek, 2018)",
        "Eutrophying emissions per kilogram (Poore & Nemecek, 2018)"
    ]]
    food_df["Entity"] = food_df["Entity"].str.lower().str.strip()

    ########### CAR ############
    fuel_df = pd.read_excel("/Users/Swastika/Downloads/ghg-conversion-factors-2024-FlatFormat_v1_1.xlsx", sheet_name='Sheet1')
    fuel_df = fuel_df[[
        "Level 3",  
        "Column Text",  
        "GHG Conversion Factor 2024",  
        "UOM"  
    ]]
    fuel_df['Column Text'] = fuel_df['Column Text'].apply(lambda x: str(x).lower() if not pd.isna(x) else 'unknown')

    car_categories = ["Mini", "Supermini", "Lower medium", "Upper medium", "Executive", "Luxury", "Sports", "Dual purpose 4X4", "MPV"]

    ########### FLIGHT ############
    flight_df = pd.read_excel("/Users/Swastika/Downloads/ghg-conversion-factors-2024-FlatFormat_v1_1.xlsx", sheet_name='Sheet2')
    flight_df = flight_df[[
        "LCA Activity",  
        "Emission Factor (kgCO₂e/passenger-km)",  
        "Description"
    ]]
    flight_df['LCA Activity'] = flight_df['LCA Activity'].astype(str).str.lower()
    flight_keywords = {"flight", "airplane", "plane", "air travel", "airline"}

    ########### BIKE ############
    df = pd.read_excel("/Users/Swastika/Downloads/ghg-conversion-factors-2024-FlatFormat_v1_1.xlsx", sheet_name='Sheet1')  
    df.columns = df.columns.str.strip()
    df = df[[
        "Level 2",  
        "Level 3",  
        "GHG Conversion Factor 2024",  
    ]]
    df["Level 2"] = df["Level 2"].astype(str).str.lower().str.strip()
    df["Level 3"] = df["Level 3"].astype(str).str.lower().str.strip()
    bike_keywords = {"bike", "motorbike", "motorcycle", "scooter"}

    ########### WASTE ############
    waste_df = pd.read_excel("/Users/Swastika/Downloads/ghg-emission-factors-hub-2025.xlsx", sheet_name='Sheet1')
    waste_df.columns = waste_df.columns.str.strip()
    waste_df["Material"] = waste_df["Material"].astype(str).str.lower().str.strip()
    waste_materials = set(waste_df["Material"].tolist())


    # Food Carbon Footprint Report
    def generate_food_footprint_report(food_items_list):
        report_data = []
        for food in food_items_list:
            match = food_df[food_df["Entity"] == food]
            if match.empty:
                best_match = process.extractOne(food, food_df["Entity"].tolist())
                if best_match and best_match[1] > 60:
                    match = food_df[food_df["Entity"] == best_match[0]]
            if not match.empty:
                report_data.append(match.iloc[0].to_dict())
        return pd.DataFrame(report_data)

    # Vehicle Carbon Footprint Report
    def generate_vehicle_report(user_input):
        detected_categories = extract_vehicle_data(user_input)
        
        if not detected_categories:
            detected_categories = ["Mini", "Supermini"]
        
        report_data = []
        for category in detected_categories:
            category_matches = fuel_df[fuel_df['Level 3'].str.contains(category, case=False, na=False)]
            for _, row in category_matches.iterrows():
                report_row = {
                    'Vehicle Type': row['Level 3'],
                    'Fuel Type': row['Column Text'],
                    'GHG Emissions per km (kg CO2e)': row['GHG Conversion Factor 2024'],
                }
                report_data.append(report_row)
        
        report_df = pd.DataFrame(report_data)
        report_df.drop_duplicates(inplace=True)
        return report_df

    # Flight Emissions Report
    def generate_flight_report(user_input):
        if flight_found:
            print("\n📊 Flight Emissions Report:\n")
            return flight_df  
        else:
            print("\nNo flight-related terms found in input.")
            return pd.DataFrame()  

    # Bike Emissions Report
    def generate_bike_report(user_input):
        """Generate emissions report if bike-related terms are detected."""
        if bike_found:
            bike_df = df[df["Level 2"].str.contains("motorbike", case=False, na=False)]
            report_df = bike_df.drop_duplicates()
            report_df.drop(columns={'Level 2'}, inplace=True)
            report_df.rename(columns={'GHG Conversion Factor 2024':'GHG Conversion Factor per km', 'Level 3':'Motorbike Size'}, inplace=True)
            if not report_df.empty:
                print("\n📊 Motorbike Emissions Report:\n")
            else:
                print("\n⚠️ No motorbike data found in the emissions table.")
            
            return report_df
        else:
            print("\n⚠️ No motorbike-related words found in input.")
            return pd.DataFrame()  

    # Waste Handling Report
    def generate_waste_report(user_input):
        extracted_waste_info = extract_waste_data(user_input)
        report_data = []
        for waste in extracted_waste_info:
            match = waste_df[waste_df["Material"].str.contains(waste, na=False, case=False)]
            
            if match.empty:
                best_match, score = process.extractOne(waste, waste_materials)
                if score > 60:  
                    match = waste_df[waste_df["Material"] == best_match]

            if not match.empty:
                for _, row in match.iterrows():
                    report_row = {
                        'Material': row['Material'].capitalize(),
                        'Recycled (kgCO₂e/short ton)': row.get('Recycled', 'N/A'),
                        'Landfilled (kgCO₂e/short ton)': row.get('Landfilled', 'N/A'),
                        'Combusted (kgCO₂e/short ton)': row.get('Combusted', 'N/A'),
                        'Composted (kgCO₂e/short ton)': row.get('Composted', 'N/A'),
                        'Anaerobically Digested (Dry Digestate with Curing) (kgCO₂e/short ton)': row.get('Anaerobically Digested (Dry Digestate with Curing)', 'N/A'),
                        'Anaerobically Digested (Wet Digestate with Curing) (kgCO₂e/short ton)': row.get('Anaerobically Digested (Wet Digestate with Curing)', 'N/A')
                    }
                    report_data.append(report_row)

        report_df = pd.DataFrame(report_data)
        report_df.drop_duplicates(inplace=True)

        if report_df.empty:
            print("\n⚠️ No matching waste materials found in the database.")
            return pd.DataFrame()
        
        return report_df


    # Streamlit UI
    st.title("Carbon Footprint Tracker")
    user_input = st.text_area("Enter your daily activities:")

    if st.button("Analyze Carbon Footprint"):
        doc = nlp(user_input.lower())
        
        food_items = [token.text for token in doc if token.text in food_df["Entity"].tolist()]

        def extract_vehicle_data(text):
            doc = nlp(text)
            vehicle_items = set()
            detected_categories = set()

            for sent in doc.sents:
                for phrase in sent.noun_chunks:
                    phrase_text = phrase.text.lower()
                    if "car" in phrase_text:
                        vehicle_items.add(phrase_text)
                        for category in car_categories:
                            if category.lower() in phrase_text:
                                detected_categories.add(category)

            return list(detected_categories)
        

        flight_found = any(token.text in flight_keywords for token in doc)

        bike_found = any(token.text in bike_keywords for token in doc)


        def extract_waste_data(text):
            doc = nlp(text.lower())
            detected_waste_items = set()
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                phrase = chunk.text.lower().strip()
                if phrase in waste_materials:
                    detected_waste_items.add(phrase)
            
            # Extract single words using lemmatization
            for token in doc:
                word = token.lemma_.lower().strip()
                if word in waste_materials:
                    detected_waste_items.add(word)

            remaining_words = [token.lemma_.lower().strip() for token in doc if token.pos_ in ('NOUN', 'ADJ') and token.lemma_.lower().strip() not in detected_waste_items]
            for word in remaining_words:
                best_match, score = process.extractOne(word, waste_materials)
                if score > 80:  # Raise the threshold significantly!
                    detected_waste_items.add(best_match)

            detected_waste_items = [item for item in detected_waste_items]

            return list(detected_waste_items)
        

        # Food Carbon Footprint Report
        with st.expander("🍽️ Food Carbon Footprint"):
            food_report = generate_food_footprint_report(food_items)
            st.dataframe(food_report)

        # Vehicle Emissions Report
        with st.expander("🚗 Vehicle Emissions"):
            vehicle_report = generate_vehicle_report(user_input)
            st.dataframe(vehicle_report)

        # Bike Emissions Report (Only if bike-related terms found)
        if bike_found:
            with st.expander("🏍️ Bike Emissions"):
                bike_report = generate_bike_report(user_input)
                st.dataframe(bike_report)

        # Flight Emissions Report (Only if flight-related terms found)
        if flight_found:
            with st.expander("✈️ Flight Emissions"):
                flight_report = generate_flight_report(user_input)
                st.dataframe(flight_report)

        # Waste Handling Report
        with st.expander("🗑️ Waste Handling Report"):
            waste_report = generate_waste_report(user_input)
            st.dataframe(waste_report)


    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    def get_gemini_recommendation(user_text):
        model = genai.GenerativeModel("gemini-1.5-flash")  # Use Gemini-Pro model
        response = model.generate_content(f"Provide personalized eco-friendly recommendations based on this user's input:\n{user_text}")
        return response.text if response else "Sorry, I couldn't generate recommendations."

    st.subheader("🔍 Personalized Sustainability Recommendations")
        
    recommendation = get_gemini_recommendation(user_input)
        
    st.write(recommendation)  # Display recommendations

