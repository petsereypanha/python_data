import pandas as pd
import re
from pathlib import Path

file_path = Path('/Users/sereypanha/Documents/CodeStudy/Data_Science/ine/clean/financials.csv')

# Define output path in the same directory as the script
output_path = file_path.parent / 'financials_cleaned.csv'

data = []
header = []
with open(file_path, 'r') as f:
    first_line = next((line for line in f if line.strip()), None)
    if not first_line:
        print("File is empty.")
    else:
        header = first_line.strip().split(None, 2)

        for line in f:
            line = line.strip()
            if line:
                row = line.split(None, 2)
                while len(row) < len(header):
                    row.append('')
                data.append(row)

df = pd.DataFrame(data, columns=header)
df.to_csv(output_path, index=False)

# Display results
print(f"File saved to: {output_path}")
print("Cleaned data preview:")
print(df.head())
print("\nDataframe Info:")
df.info()