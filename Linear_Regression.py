###############################################################################
# Life Insurance Premium Prediction - Linear Regression
###############################################################################           

###############################################################################
# Import Required Packages
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score

###############################################################################
# Import Sample Data
###############################################################################

# Import
data_for_model = pd.read_excel('Life insurance.xlsx', sheet_name='Life insurance')  

# Shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)               

###############################################################################
# Get Data Information
###############################################################################

data_for_model.info()

data_for_model['region'].value_counts()

###############################################################################
# Deal with Missing Values
###############################################################################

data_for_model.isna().sum()     

###############################################################################
# Deal with Outliers
###############################################################################

data_for_model.describe()

data_for_model.plot(kind='box', subplots = True, layout = (3, 5))

###############################################################################
# Split Input Variables and Output Variables
###############################################################################

X = data_for_model.drop(['premium'], axis = 1)
y = data_for_model['premium']

###############################################################################
# Split out Training and Test Sets
###############################################################################

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42)

###############################################################################
# Deal with Categorical Variables
###############################################################################

# Create a list of categorical variables                    
categorical_vars = ['sex', 'smoker', 'region', 'CI', 'rated', 'UL permanent', 'disability']   

# Create and apply OneHotEncoder while removing the dummy variable
one_hot_encoder = OneHotEncoder(sparse = False, drop = 'first')               

# Apply fit_transform on training data
X_train_encoded =  one_hot_encoder.fit_transform(X_train[categorical_vars])

# Apply transform on test data
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])            

# Get feature names to see what each column in the 'encoder_vars_array' presents
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Convert our result from an array to a DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)

# Concatenate (Link together in a series or chain) new DataFrame to our original DataFrame 
X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1)    
X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1)    
 
# Drop the original categorical variable columns
X_train.drop(categorical_vars, axis = 1, inplace = True)           
X_test.drop(categorical_vars, axis = 1, inplace = True) 

###############################################################################
# Data Visualization
###############################################################################

# Data Distribution
data_for_model.hist(figsize=(10,8))  

# Pairplot
sns.pairplot(data_for_model)

###############################################################################
# Feature Scaling
###############################################################################

scale_standard = StandardScaler()

X_train = pd.DataFrame(scale_standard.fit_transform(X_train), columns = X_train.columns)        
X_test = pd.DataFrame(scale_standard.transform(X_test), columns = X_test.columns)   

###############################################################################
# Model Training
###############################################################################

regressor = LinearRegression()
regressor.fit(X_train, y_train)

###############################################################################
# Prediction
###############################################################################

# Predict on the test set
y_pred = regressor.predict(X_test)

###############################################################################
# Model Assessment (Validation)
###############################################################################

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)

# Calculate adjusted R-squared
num_data_points, num_input_vars = X_test.shape                           
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

# Cross validation (KFold: including both shuffling and the random state)
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)    
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = 'r2')     # returns r2 for each chunk of data (each cv)
cv_scores.mean()

# Extract model coefficients
coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names, coefficients], axis = 1)
summary_stats.columns = ['input_variable', 'coefficient']

# Extract model intercept
regressor.intercept_

###############################################################################
# Predict on New Data
###############################################################################

# Create new data

new_data = pd.DataFrame({'age': [41, 29, 41],
                         'sex': ['male', 'male', 'female'],
                         'bmi': [20, 30, 20],
                         'children': [1, 0 ,1],
                         'smoker': ['no', 'yes', 'no'],
                         'region': ['GTA', 'GTA', 'GTA'], 
                         'CI': ['yes', 'no', 'yes'],
                         'rated': ['no', 'no', 'no'],
                         'UL permanent': ['no', 'no', 'yes'],
                         'disability': ['no', 'no', 'no']})       

# Apply the same transformations to new data

new_data_encoded = one_hot_encoder.transform(new_data[categorical_vars])
new_data_encoded = pd.DataFrame(new_data_encoded, columns=encoder_feature_names)
new_data = pd.concat([new_data.reset_index(drop=True), new_data_encoded.reset_index(drop=True)], axis=1)
new_data.drop(categorical_vars, axis=1, inplace=True)
new_data = pd.DataFrame(scale_standard.transform(new_data), columns=new_data.columns)


# Pass new data in and receive predictions

new_predictions = regressor.predict(new_data)
print(new_predictions)