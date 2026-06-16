import pandas as pd
import folium

# Replace this with the path to your Excel file
excel_file = r'D:\brainyAI\lucknow_running_7_july_windows\converted_file_GD-BHR.xlsx'

# Read the Excel file
df = pd.read_excel(excel_file, engine='openpyxl')

# Function to convert DDM to DD
def convert_ddm_to_dd(coord):
    degrees = int(coord / 100)
    minutes = coord - (degrees * 100)
    return degrees + (minutes / 60)

# Apply conversion
df['Lat'] = df['Lat'].apply(convert_ddm_to_dd)
df['Lon'] = df['Lon'].apply(convert_ddm_to_dd)

# Create folium map centered on mean location
center_lat = df['Lat'].mean()
center_lon = df['Lon'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add markers
for idx, row in df.iterrows():
    folium.Marker(
        location=[row['Lat'], row['Lon']],
        popup=row['location'],
        tooltip=row['location']
    ).add_to(m)

name = "map_gonda"
# Save map to HTML file
m.save(f'{name}.html')
print(f"Map has been saved to '{name}.html'. Open it in your browser to view.")
