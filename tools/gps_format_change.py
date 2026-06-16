import pandas as pd

# Load your Excel file (update path as needed)
input_file = r'D:\brainyAI\lucknow_running_7_july_windows\Gonda_Behrai_file.xlsx'
output_file = r'converted_file_GD-BHR.xlsx'

# Read the Excel file
df = pd.read_excel(input_file)

# Define conversion function: DD → DDM
def convert_dd_to_ddm(dd):
    degrees = int(dd)
    minutes = (abs(dd) - abs(degrees)) * 60
    return degrees * 100 + minutes

# Apply to latitude and longitude columns
df['Lat'] = df['Lat'].apply(convert_dd_to_ddm)
df['Lon'] = df['Lon'].apply(convert_dd_to_ddm)

# Save to a new Excel file
df.to_excel(output_file, index=False)
print(f"Converted file saved to: {output_file}")
