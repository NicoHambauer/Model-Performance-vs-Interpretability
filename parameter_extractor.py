import os
import pandas as pd

# Specify the folder containing the subfolders with csv files
folder_path = "./results_bise"

# Initialize empty DataFrame
df_params = pd.DataFrame(columns=["Model Type", "Dataset Name", "Parameter Strings"])

# Iterate over sub-folders in main folder
for sub_folder in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, sub_folder)
    if os.path.isdir(sub_folder_path):
        # Extract model type from sub-folder name
        model_type = sub_folder.split("_")[-1]
        # Iterate over CSV files in sub-folder
        for file_name in os.listdir(sub_folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(sub_folder_path, file_name)
                # Read CSV file into DataFrame
                df = pd.read_csv(file_path)
                # Iterate over rows in DataFrame
                for idx, row in df.iterrows():
                    # Extract dataset name and parameter string
                    dataset_name = row[0]

                    # check if row[1] is not nan
                    if not isinstance(row[1], str):
                        continue
                    param_str = row[1].replace("\n", "")
                    # Append row to df_params
                    df_params = df_params.append({"Model Type": model_type, "Dataset Name": dataset_name, "Parameter Strings": param_str}, ignore_index=True)

# Group df_params by Model Type and Dataset Name
df_params = df_params.groupby(["Model Type", "Dataset Name"], as_index=False).agg({"Parameter Strings": lambda x: "\n\n".join(x)})

# Pivot df_params and format as desired
df_params = df_params.pivot(index="Dataset Name", columns="Model Type", values="Parameter Strings").reset_index().rename_axis(None, axis=1)

# Print resulting DataFrame
print(df_params)

# Save the grouped parameters to a csv file
df_params.to_csv("grouped_parameters.csv", index=False)
