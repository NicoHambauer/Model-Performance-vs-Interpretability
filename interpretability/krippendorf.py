import krippendorff
import numpy as np
import pandas as pd

# Load the data
data = pd.read_csv('KrippAlpha_Round2.csv', delimiter=';', header=None)

# Replace missing values represented by '-'
data = data.replace({'-': np.nan})

# Split the data into blocks by annotator
blocks = np.split(data, data[data.isna().all(axis=1)].index)

# Remove the empty rows and header rows within each block
annotators = []
for block in blocks:
    block = block.dropna(how='all')
    if not block.empty:
        annotator_name = block.iloc[0, 0]
        block = block.iloc[1:, 1:]
        # Convert all data to float
        for col in block.columns:
            block[col] = block[col].apply(lambda x: float(x) if pd.notnull(x) else np.nan)
        annotators.append((annotator_name, block))

# Prepare reliability data
reliability_data = [annotator[1].values.flatten().tolist() for annotator in annotators]

# Calculate Krippendorff's alpha
alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="ordinal")

print(f"Krippendorff's alpha: {alpha:.4f}")