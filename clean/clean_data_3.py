import pandas as pd
import re
from pathlib import Path

file_path = Path('/Users/sereypanha/Documents/CodeStudy/Data_Science/ine/clean/financials.csv')
output_path = file_path.parent / 'financials_cleaned.csv'

cleaned_lines = []

with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # If the line does not contain a comma, it's space-delimited
        if ',' not in line:
            # Replace 2 or more spaces with a single comma
            cleaned_line = re.sub(r'\s{2,}', ',', line)
            cleaned_lines.append(cleaned_line)
        else:
            # Otherwise, it's already comma-separated, so add it as is
            cleaned_lines.append(line)

# Check if we have data
if not cleaned_lines:
    raise ValueError("File is empty or has no valid data")

# Parse header from first line
header = cleaned_lines[0].split(',')
print(f"Header: {header}")

# Parse data from remaining lines
data = []
for line in cleaned_lines[1:]:
    row = line.split(',')
    data.append(row)

# Create DataFrame and save
df = pd.DataFrame(data, columns=header)
df.to_csv(output_path, index=False)

print(f"File saved to: {output_path}")
print("Cleaned data preview:")
print(df.head())
print("\nDataframe Info:")
df.info()