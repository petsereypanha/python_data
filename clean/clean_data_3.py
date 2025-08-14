import pandas as pd
import re
from pathlib import Path

file_path = Path('/Users/sereypanha/Documents/CodeStudy/Data_Science/ine/clean/inv_sep.csv')
output_path = file_path.parent / 'inv_sep_cleaned.csv'

cleaned_lines = []
data = []
header_found = False

with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if not header_found and ('cid' in line and 'invoice_date' in line and 'bill_ctry' in line):
            header_found = True
            if ',' in line:
                header = [h.strip() for h in line.split(',')]
            else:
                clean_header_line = re.sub(r',', ' ', line)
                clean_header_line = re.sub(r'\s{2,}', ' ', clean_header_line)
                header = clean_header_line.strip().split(' ')
            continue

        if header_found:
            num_columns = len(header)
            row = line.split(None, num_columns - 1)

            if len(row) == num_columns:
                data.append(row)

# Create DataFrame and save
df = pd.DataFrame(data, columns=header)
df.to_csv(output_path, index=False)

print(f"File saved to: {output_path}")
print("Cleaned data preview:")
print(df.head())
print("\nDataframe Info:")
df.info()