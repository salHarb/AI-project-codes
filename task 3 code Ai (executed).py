import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to perform linear regression
def perform_linear_regression(X, y):
    # Convert numpy arrays to DataFrames
    X_df = pd.DataFrame(X,
                        columns=['Group', 'Sentence_ID', 'TRIAL_INDEX', 'CURRENT_FIX_X', 'CURRENT_FIX_Y', 'NEXT_FIX_X',
                                 'NEXT_FIX_Y', 'CURRENT_FIX_DURATION', 'NEXT_FIX_DURATION'])
    y_df = pd.DataFrame(y, columns=['Reading_speed'])



    # Create linear regression object
    model = LinearRegression()

    # Fit the model
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Add predicted values to the DataFrame
    X_df['Predicted_Reading_speed'] = y_pred

    # Print the dataset after regression

    print(X_df)

# Read data from Excel files
def read_excel_data(path):
    return pd.read_excel(path)

# Paths to Excel files
demo_path = r"C:\Ai 12th project\datasets\demo.xlsx"
fix_stats_path = r"C:\Ai 12th project\datasets\Fix_stats.xlsx"
fixation_report_path = r'C:\Ai 12th project\datasets\Fixation_report.xlsx'

# Read data from Excel files
demo_data = read_excel_data(demo_path)
fix_stats_data = read_excel_data(fix_stats_path)
fixation_report_data = pd.read_excel(fixation_report_path)

# Drop non-numeric columns for demo data
X_demo = demo_data.drop(['SubjectID', 'Reading_speed'], axis=1)
y_demo = demo_data['Reading_speed']

# Drop non-numeric columns for Fix_stats data
X_fix_stats = fix_stats_data.drop(['SubjectID'], axis=1)
y_fix_stats = fix_stats_data['FIX_DURATION_mean']

# Drop non-numeric columns for Fix_stats data
X_fix_stats = fix_stats_data.drop(['SubjectID'], axis=1)
y_fix_stats = fix_stats_data['FIX_DURATION_mean']



# Impute missing values in Fix_stats data using mean imputation
imputer = SimpleImputer(strategy='mean')
X_fix_stats_imputed = imputer.fit_transform(X_fix_stats)

# Perform linear regression for demo data
print("Demo Data Linear Regression:")
perform_linear_regression(X_demo, y_demo)

# Encode categorical variables if necessary
encoder = LabelEncoder()
X_demo['Group'] = encoder.fit_transform(X_demo['Group'])

# Perform linear regression for Fix_stats data with imputed values
print("\nFix_stats Data Linear Regression:")
perform_linear_regression(X_fix_stats_imputed, y_fix_stats)

# Step 1: Read dataset from Excel using path
df = fixation_report_data

# Display the first few rows of the dataset
print("\nFix_reports Data Linear Regression:")
print(df.head())

# Step 2: Extracting independent and dependent variables
X = df[['Sentence_ID', 'Word_Number']]  # Independent variables: Sentence_ID and Word_Number
y = df['FIX_DURATION']  # Dependent variable: FIX_DURATION

# Step 3: Create a Linear Regression model and fit the data
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict y-values using the model
y_pred = model.predict(X)

# Step 5: Visualize the data and the regression line (Since it's multi-dimensional, visualization may not be straightforward)

# Step 6: Print the coefficients
print("\nIntercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Load the dataset from Excel file
excel_file_path = r'C:\Ai 12th project\datasets\IA_report.xlsx'
df = pd.read_excel(excel_file_path)

# Define features and target variable
features = ['Group', 'Sentence_ID', 'Word_Number', 'FIXATION_COUNT', 'SKIP',
            'TOTAL_READING_TIME', 'FIRST_FIXATION_DURATION', 'FIRST_FIXATION_X', 'FIRST_FIXATION_Y',
            'FIRST_RUN_TOTAL_READING_TIME', 'FIRST_SACCADE_AMPLITUDE', 'REGRESSION_IN',
            'REGRESSION_OUT', 'REGRESSION_OUT_FULL', 'REGRESSION_PATH_DURATION']
target = 'QUESTION_ACCURACY'

X = df[features]  # Features
y = df[target]    # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Combine actual and predicted values with features
output_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
output_df = pd.concat([X_test, output_df], axis=1)

# Print the dataset after linear regression
print("\nAI data Linear Regression:")
print(output_df)
