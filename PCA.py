import pandas as pd
from sklearn.decomposition import PCA

# Define paths to Excel files
paths = [
    "C:\\Ai 12th project\\datasets\\demo.xlsx",
    "C:\\Ai 12th project\\datasets\\Fix_stats.xlsx",
    "C:\\Ai 12th project\\datasets\\Fixation_report.xlsx",
    "C:\\Ai 12th project\\datasets\\IA_report.xlsx"
]

# Iterate over paths
for path in paths:
    print("Reading dataset from:", path)
    try:
        # Read dataset from Excel file
        df = pd.read_excel(path)

        # Get feature column names
        feature_columns = df.columns.tolist()

        # Remove non-numeric columns
        non_numeric_columns = ["Group", "SubjectID"]
        feature_columns = [col for col in feature_columns if col not in non_numeric_columns]

        # Convert non-numeric values to NaN and drop rows with NaN values
        df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)

        # Separate features
        X = df[feature_columns]

        # Perform PCA
        pca = PCA()
        pca.fit(X)

        # Transform features using PCA
        components = pca.transform(X)
        component_names = [f"Component_{i + 1}" for i in range(components.shape[1])]

        # Append transformed components to the dataset
        component_df = pd.DataFrame(components, columns=component_names)
        df = pd.concat([df, component_df], axis=1)

        # Print the dataset after PCA
        print("Dataset after PCA:")
        print(df)
        print()

    except Exception as e:
        print("Error reading dataset:", e)
