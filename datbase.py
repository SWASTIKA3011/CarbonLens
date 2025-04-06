# import streamlit as st
# import sqlite3
# import io
# import base64
# import pandas as pd
# from PIL import Image, UnidentifiedImageError

# # Connect to SQLite database
# DB_PATH = "/Users/swastika/Carbon Footprint App/peatland_requests.db"
# conn = sqlite3.connect(DB_PATH)
# cursor = conn.cursor()

# # Fetch data
# cursor.execute("SELECT id, name, phone, email, id_proof FROM peatland_requests")
# rows = cursor.fetchall()

# df = pd.DataFrame(rows, columns=["ID", "Name", "Phone", "Email", "ID Proof"])

# # Function to convert BLOB to Base64 Image
# def get_image_html(blob_data):
#     if blob_data:
#         try:
#             image = Image.open(io.BytesIO(blob_data))  # Try to open as image
#             image.thumbnail((100, 100))  # Resize for display
#             buf = io.BytesIO()
#             image.save(buf, format="PNG")  # Convert to PNG
#             base64_str = base64.b64encode(buf.getvalue()).decode()
#             return f'<img src="data:image/png;base64,{base64_str}" width="100" height="100">'
#         except UnidentifiedImageError:
#             return "⚠️ Not an Image (Possible PDF)"
#     return "No Image"

# # Generate HTML Table
# table_html = "<table border='1'><tr><th>ID</th><th>Name</th><th>Phone</th><th>Email</th><th>ID Proof</th></tr>"

# for _, row in df.iterrows():
#     image_html = get_image_html(row["ID Proof"])
#     table_html += f"<tr><td>{row['ID']}</td><td>{row['Name']}</td><td>{row['Phone']}</td><td>{row['Email']}</td><td>{image_html}</td></tr>"

# table_html += "</table>"

# # Display Table
# st.markdown("<h3>📋 Peatland Requests Data</h3>", unsafe_allow_html=True)
# st.markdown(table_html, unsafe_allow_html=True)

# conn.close()

# ---new---
import streamlit as st
import sqlite3
import io
import base64
import pandas as pd
from PIL import Image, UnidentifiedImageError

# Connect to SQLite database
DB_PATH = "/Users/swastika/Carbon Footprint App/peatland_requests.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Fetch all column names dynamically
cursor.execute("PRAGMA table_info(peatland_requests)")
columns = [col[1] for col in cursor.fetchall()]  # Extract column names

# Fetch data dynamically
cursor.execute(f"SELECT {', '.join(columns)} FROM peatland_requests")
rows = cursor.fetchall()

# Convert to DataFrame
df = pd.DataFrame(rows, columns=columns)

# Function to convert BLOB to Base64 Image
# def get_image_html(blob_data):
#     if blob_data:
#         try:
#             image = Image.open(io.BytesIO(blob_data))  # Try to open as image
#             print(f"Original Image Mode: {image.mode}") 
#             image.thumbnail((100, 100))  # Resize for display
#             buf = io.BytesIO()

#             # 🛠️ Convert to RGB (force mode correction)
#             if image.mode not in ("RGB", "RGBA"):
#                 image = image.convert("RGB")

#             # 🛠️ Handle multi-page TIFFs (select first page)
#             if getattr(image, "n_frames", 1) > 1:
#                 image.seek(0)

#             image.save(buf, format="PNG")  # Convert to PNG
#             base64_str = base64.b64encode(buf.getvalue()).decode()
#             return f'<img src="data:image/png;base64,{base64_str}" width="100" height="100">'
#         except UnidentifiedImageError:
#             return "⚠️ Not an Image (Possible PDF)"
#     return "No Data"
def get_image_html(blob_data):
    if blob_data:
        try:
            image = Image.open(io.BytesIO(blob_data))  # Open as image
            print(f"Original Image Mode: {image.mode}")  # Debug mode

            # 🛠️ Convert 16-bit grayscale (I;16) to 8-bit grayscale (L) or RGB
            if image.mode == "I;16":
                image = image.convert("I")  # Convert to 32-bit integer grayscale
                image = image.point(lambda p: p * (255.0 / 65535.0))  # Normalize to 8-bit
                image = image.convert("L")  # Convert to 8-bit grayscale
                image = image.convert("RGB")  # Finally, convert to RGB

            # 🛠️ Handle multi-page JP2 (if applicable)
            if getattr(image, "n_frames", 1) > 1:
                image.seek(0)

            # Resize for display
            image.thumbnail((100, 100))

            # Save as PNG for embedding
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            base64_str = base64.b64encode(buf.getvalue()).decode()
            return f'<img src="data:image/png;base64,{base64_str}" width="100" height="100">'
        
        except UnidentifiedImageError:
            return "⚠️ Not an Image (Possible PDF)"
        except ValueError as e:
            return f"⚠️ Image Processing Error: {str(e)}"
        except Exception as e:
            return f"⚠️ Error: {str(e)}"
    
    return "No Data"


# Generate HTML Table
table_html = "<table border='1'><tr>"

# Add dynamic headers
for column in columns:
    table_html += f"<th>{column}</th>"
table_html += "</tr>"

# Populate table rows
for _, row in df.iterrows():
    table_html += "<tr>"
    for column in columns:
        value = row[column]
        
        # Convert images if it's the "id_proof" or any image-related column
        if "image" in column.lower() or "id_proof" in column.lower():
            table_html += f"<td>{get_image_html(value)}</td>"
        else:
            table_html += f"<td>{value if value is not None else 'No Data'}</td>"
    
    table_html += "</tr>"

table_html += "</table>"

# Display Table
st.markdown("<h3>📋 Peatland Requests Data</h3>", unsafe_allow_html=True)
st.markdown(table_html, unsafe_allow_html=True)

conn.close()
