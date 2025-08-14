import pandas as pd
import re
from pathlib import Path

file_path = Path('/Users/sereypanha/Documents/CodeStudy/Data_Science/ine/clean/crews.csv')

output_path = file_path.parent / 'crews_cleaned.csv'

data = []
header = []

with open(file_path, 'r') as f:
    lines = [line for line in f if line.strip()]
    # Check if file has data
    if not lines:
        raise ValueError("File is empty or has no valid data")

    # The first non-empty line is the header
    header_line = lines[0].strip()

    header_match = re.match(r'^(\S+)\s+(\S+)\s+(\S+)\s+(.*)$', header_line)
    if header_match:
        header = [header_match.group(1), header_match.group(2), header_match.group(3), header_match.group(4)]
    else:
        # Fallback: split by whitespace and take first 4 parts
        header_parts = header_line.split()
        header = header_parts[:4] if len(header_parts) >= 4 else ['col1', 'col2', 'col3', 'col4']

    print(f"Header: {header}")

    data_lines = lines[1:]

    for line in data_lines:
        match = re.match(r'^(\d+)\s+(\d+)\s+(.*?)\s\s+(.*)$', line)
        if match:
            row = [match.group(1), match.group(2), match.group(3), match.group(4)]
            data.append(row)

df = pd.DataFrame(data, columns=header)
df.to_csv(output_path, index=False)

print(f"File saved to: {output_path}")
print("Cleaned data preview:")
print(df.head())
print("\nDataframe Info:")
df.info()