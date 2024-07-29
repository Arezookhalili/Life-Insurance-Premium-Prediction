###############################################################################
# Life Insurance Premium Prediction - Decision Tree
###############################################################################


###############################################################################
# Import Required Packages
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree

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
# Model Training
###############################################################################

regressor = DecisionTreeRegressor(random_state = 42, max_depth = 4)
     # regressor = DecisionTreeRegressor(random_state = 42, max_depth = 3)       # To refit the model based on the refit explanation section

regressor.fit(X_train, y_train)

###############################################################################
# Prediction
###############################################################################

# Predict on the test set
y_pred = regressor.predict(X_test)

###############################################################################
# Model Assessment (Validation)
###############################################################################

# First approach: Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)

# Calculate adjusted R-squared
num_data_points, num_input_vars = X_test.shape                           
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

# Cross validation (KFold: including both shuffling and the random state)
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)                    # n_splits: number of equally sized chunk of data
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = 'r2')
cv_scores.mean()

# Finding the best max depth

max_depth_list = list(range(1,9))
accuracy_scores = []

for depth in max_depth_list:
    
    regressor = DecisionTreeRegressor(max_depth = depth, random_state = 42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test,y_pred)
    accuracy_scores.append(accuracy)
    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

# Plot of max depths
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = 'x', color = 'red')
plt.title(f'Accuracy by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy, 4)})')
plt.xlabel('Max Depth of Decision Tree')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

"""
best max depth is 3 but refitting the model with max depth of 3 did not change the validation score.
"""

# Plot our model

plt.figure(figsize=(35,15))
tree = plot_tree(regressor,
                 feature_names = X_train.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 24)

###############################################################################
# Predict on New Data
###############################################################################

# Create new data

new_data = pd.DataFrame({'age': [41, 29, 41],
                         'sex': ['male', 'male', 'female'],
                         'bmi': [20, 30, 20],
                         'children': [1, 0 ,1],
                         'smoker': ['no', 'no', 'no'],
                         'region': ['GTA', 'GTA', 'GTA'], 
                         'CI': ['yes', 'no', 'yes'],
                         'rated': ['no', 'no', 'no'],
                         'UL permanent': ['no', 'no', 'no'],
                         'disability': ['no', 'no', 'no']})

# Apply the same transformations to new data

new_data_encoded = one_hot_encoder.transform(new_data[categorical_vars])
new_data_encoded = pd.DataFrame(new_data_encoded, columns=encoder_feature_names)
new_data = pd.concat([new_data.reset_index(drop=True), new_data_encoded.reset_index(drop=True)], axis=1)
new_data.drop(categorical_vars, axis=1, inplace=True)


# Pass new data in and receive predictions

new_predictions = regressor.predict(new_data)
print(new_predictions)


