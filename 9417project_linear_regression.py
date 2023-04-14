import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

data = pd.read_csv('mini_train.csv')

data["hour"] = pd.to_datetime(data['timestamp']).dt.hour
data["weekday"] = pd.to_datetime(data['timestamp']).dt.weekday
data = data.drop("timestamp", axis=1)

data_update = data.dropna(thresh=len(data)*0.6, axis=1)


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(data_update['primary_use'])


encoded_col = encoder.transform(data_update['primary_use'])
data_update['primary_use'] = encoded_col





# Remove outliers
Q1 = data_update.quantile(0.25)
Q3 = data_update.quantile(0.75)
IQR = Q3 - Q1
data_cleaned = data_update[~((data_update < (Q1 - 1.5 * IQR)) | (data_update > (Q3 + 1.5 * IQR))).any(axis=1)]




X = data_cleaned.drop(columns = ['meter_reading'])




Y= data_cleaned[['meter_reading']]




#Fill in the null values with the most frequent values in column 
X_filled = X.fillna(X.mode().iloc[0])




X_filled_train, X_filled_test, Y_train, Y_test = train_test_split(X_filled, Y, test_size=0.2, random_state=42)




from sklearn.preprocessing import StandardScaler
# Create a StandardScaler object
scaler = StandardScaler()
X_filled_train_scaled = scaler.fit_transform(X_filled_train)
X_filled_test_scaled = scaler.transform(X_filled_test)




model = LinearRegression()
# Train the model on the training data
model.fit(X_filled_train_scaled, Y_train)
# Use the model to make predictions on the testing data
y_pred = model.predict(X_filled_test_scaled)




from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, y_pred)
print('Mean squared error:', mse)
r2 = r2_score(Y_test, y_pred)
print('R-squared:', r2)
rmse = mean_squared_error(Y_test, y_pred, squared=False)
print('Root mean squared error:', rmse)




