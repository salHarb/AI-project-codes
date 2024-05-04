import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Define paths to Excel files
paths = [
    "C:\\Ai 12th project\\datasets\\demo.xlsx",
    "C:\\Ai 12th project\\datasets\\Fix_stats.xlsx",
    "C:\\Ai 12th project\\datasets\\Fixation_report.xlsx",
    "C:\\Ai 12th project\\datasets\\IA_report.xlsx"
]

# Define target and feature column names for each dataset
target_feature_mapping = {
    "demo.xlsx": (
    "Reading_speed", ["Group", "SubjectID", "Sex", "Grade", "Age", "IQ", "Sound_detection", "Sound_change"]),
    "Fix_stats.xlsx": ("FIX_DURATION_mean",
                       ["Group", "SubjectID", "Sentence_ID", "Word_Number", "FIX_X_mean", "FIX_X_std", "FIX_Y_mean",
                        "FIX_Y_std"]),
    "Fixation_report.xlsx": ("FIX_DURATION", ["Group", "SubjectID", "Sentence_ID", "Word_Number", "FIX_X", "FIX_Y"]),
    "IA_report.xlsx": ("TOTAL_READING_TIME",
                       ["Group", "SubjectID", "Sentence_ID", "Word_Number", "QUESTION_ACCURACY", "FIXATION_COUNT",
                        "SKIP", "FIRST_FIXATION_DURATION", "FIRST_FIXATION_X", "FIRST_FIXATION_Y",
                        "FIRST_RUN_TOTAL_READING_TIME", "FIRST_SACCADE_AMPLITUDE", "REGRESSION_IN", "REGRESSION_OUT",
                        "REGRESSION_OUT_FULL", "REGRESSION_PATH_DURATION"])
}

# Iterate over paths
for path in paths:
    print("Reading dataset from:", path)
    try:
        # Read dataset from Excel file
        df = pd.read_excel(path)

        # Get target and feature column names
        target_column, feature_columns = target_feature_mapping.get(path.split("\\")[-1], (None, []))

        if target_column is None or not feature_columns:
            print("Target and feature columns not found for", path)
            continue

        # Convert non-numeric values to NaN and drop rows with NaN values
        df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)

        # Check if there are samples available for regression
        if df.shape[0] == 0:
            print("No samples available for regression in", path)
            continue

        # Separate target and features
        X = df[feature_columns]
        y = df[target_column]

        # Perform Decision Tree regression
        model = DecisionTreeRegressor()
        model.fit(X, y)

        # Print the dataset after decision tree regression
        print("Dataset after decision tree regression:")
        print(df)
        print()

    except Exception as e:
        print("Error reading dataset:", e)
