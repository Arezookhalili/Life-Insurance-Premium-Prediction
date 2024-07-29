---
layout: post
title: Predicting Life Insurance Premiums for Clients
image: "/posts/classification-title-img.png"
tags: [Machine Learning, Regression, Python]
---

In the rapidly evolving financial services industry, companies strive to provide accurate and personalized offerings to their clients. An insurance agency working with a leading financial services provider, embarked on a mission to enhance their life insurance offerings by leveraging the power of machine learning. 
This initiative aimed to design a platform that would give clients access to predict premiums for a combined life insurance and investment package with greater accuracy, ensuring fair pricing and personalized plans. 

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

The agency serves a broad spectrum of clients, each with unique financial needs and health profiles. Traditionally, calculating life insurance premiums involved a complex evaluation of multiple factors, often leading to discrepancies and inefficiencies. While financial advisors have access to software for evaluating different plans and determining premiums, the agency wants to empower clients with a platform that allows them to get a rough estimate of their premiums based on their specific budget and status, as well as the potential savings.

### Actions <a name="overview-actions"></a>

To tackle this challenge, a comprehensive data-driven approach was adopted. The journey began with the collection of extensive client data, including demographic information, health metrics, and lifestyle factors. This data was then meticulously preprocessed to ensure its accuracy and completeness.

I built a predictive model to find relationships between client metrics and *life insurance premium* for previous clients, and used this to predict premiums for potential new clients.

As I was predicting a numeric output, I tested three regression modeling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest
<br>
<br>

### Results <a name="overview-results"></a>

The Random Forest had the highest predictive accuracy.

<br>
**Metric 1: R-Squared (Test Set)**

* Random Forest = 0.918
* Decision Tree = 0.908
* Linear Regression = 0.795
  
<br>
**Metric 2: Adjusted R-Squared (Test Set)**

* Random Forest = 0.915
* Decision Tree = 0.904
* Linear Regression = 0.786

<br>
**Metric 3: Cross Validated R-Squared (K-Fold Cross Validation, k = 4)**

* Random Forest = 0.881
* Decision Tree = 0.865
* Linear Regression = 0.743

As the most important outcome for this project was predictive accuracy, rather than explicitly understanding weighted drivers of prediction, I chose the Random Forest as the model to use for making predictions on the life insurance premiums for future clients.
<br>
<br>

### Key Definition  <a name="overview-definition"></a>

age: client's age
sex: client's gender
bmi: Body mass index is a value derived from the mass and height of a person (the body mass (kg) divided by the square of the body height (m^2))
children: number of client's children
smoker: client's smoking status	
region: client's place of residence
CI: includes critical illness insurance
rated: increased premium due to health problems
UL permanent: combined investment and life insurance
disability: includes disability insurance
premium: client's monthly payment

___

# Data Overview  <a name="data-overview"></a>

The initial dataset included various attributes such as age, gender, BMI, number of children, smoking status, region, and several insurance-related features. To prepare the data for modeling, several key steps were undertaken:

Handling Missing Values: Any missing values in the dataset were identified and appropriately addressed.

Dealing with Outliers: The dataset was examined for outliers to ensure the integrity of the data.

Encoding Categorical Variables: Categorical variables like gender, smoker status, and region were encoded using one-hot encoding to make them suitable for machine learning models.

Feature Scaling: Numerical features were standardized to ensure they were on a comparable scale, enhancing the model's performance.

# Model Training and Evaluation  <a name="Model Training and Evaluation"></a>

With the data prepared, the next step was to train a machine learning model capable of accurately predicting life insurance premiums.

I tested three regression modeling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest

For each model, I imported the data in the same way but needed to pre-process the data based on the requirements of each particular algorithm. I trained & tested each model, refined each to provide optimal performance, and then measured this predictive performance based on several metrics to give a well-rounded overview of which is best.

The dataset was split into training and testing sets, ensuring that the model could be evaluated on unseen data. The model was trained on the training set, and its performance was evaluated using the testing set. Key metrics of R-squared and adjusted R-squared were calculated to assess the model's accuracy.

To further refine the model, cross-validation was performed using KFold, providing a more robust evaluation by splitting the data into multiple folds and ensuring the model's consistency across different subsets of the data.










Optimal Model Selection
Determining the optimal complexity of the decision tree was crucial. By experimenting with different maximum depths for the tree, the optimal depth was identified based on the highest accuracy score. This step ensured that the model was neither too simple to capture essential patterns nor too complex to overfit the training data.

Results and Insights
The final decision tree model provided valuable insights into the factors influencing life insurance premiums. Feature importance analysis highlighted the key variables impacting premium calculations, offering transparency and interpretability to WFG's underwriting process.

Visualization and Predictions
Visualizations such as histograms, pair plots, and tree plots were created to understand the data distribution and model structure better. Additionally, the model was used to predict premiums for new clients, showcasing its practical applicability in real-world scenarios.

Impact and Future Directions
The implementation of this machine learning solution marked a significant milestone for WFG. By accurately predicting life insurance premiums, WFG was able to offer fairer and more personalized insurance plans to their clients, enhancing customer satisfaction and trust.

Looking ahead, WFG plans to continuously refine and expand this model by incorporating additional data sources and exploring more advanced machine learning techniques. This initiative represents a commitment to innovation and excellence, ensuring that WFG remains at the forefront of the financial services industry.

Through this project, WFG has demonstrated the transformative power of data and machine learning in revolutionizing traditional financial processes, paving the way for a more efficient and customer-centric future.



Growth:
It would also allow clients to play with premiums and see how different premiums could allocate money for their retirements.
