import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import gdown


file_id = '1rdedCa3-fiF4vwLvE5BRkE1oCTZFHPsu'
url = f'https://drive.google.com/uc?id={file_id}'


output = 'listingsair.csv'
gdown.download(url, output, quiet=False)


data = pd.read_csv(output)

columns = ['price', 'neighbourhood_cleansed', 'room_type', 'accommodates', 
           'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'minimum_nights',
           'number_of_reviews', 'reviews_per_month', 'review_scores_rating', 
           'availability_365']
data = data[columns]

data['bathrooms_text'] = data['bathrooms_text'].fillna('1 bath')
data['bedrooms'] = data['bedrooms'].fillna(1)
data['beds'] = data['beds'].fillna(1)
data['review_scores_rating'] = data['review_scores_rating'].fillna(data['review_scores_rating'].mean())
data['reviews_per_month'] = data['reviews_per_month'].fillna(0)

data['price'] = data['price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

data['bathrooms_text'] = data['bathrooms_text'].str.extract('(\d+)').astype(float)

categorical_features = ['neighbourhood_cleansed', 'room_type']
numerical_features = ['accommodates', 'bathrooms_text', 'bedrooms', 'beds', 'minimum_nights', 
                      'number_of_reviews', 'reviews_per_month', 'review_scores_rating', 'availability_365']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}


X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


results = {}
for model_name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    results[model_name] = {'MSE': mse, 'MAPE': mape}


results_df = pd.DataFrame(results).T
print(results_df)


best_model = RandomForestRegressor()
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', best_model)])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)


mse = mean_squared_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)

print(f"Final model evaluation:\nMSE: {mse}\nMAPE: {mape}")


comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(comparison_df.head())


plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.3)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
