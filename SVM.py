import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# load the dataset
train_data = pd.read_csv('train.csv')

# select the features and target variable
X = train_data.drop(['Id', 'SalePrice'], axis=1)
y = train_data['SalePrice']

# fill missing values with the mean
X.fillna(X.mean(), inplace=True)

# one-hot encode categorical variables
X = pd.get_dummies(X)

# standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the SVM model
svm_model = SVR(kernel='linear', C=100, gamma='auto')
svm_model.fit(X_train, y_train)

# make predictions on the test set
y_pred = svm_model.predict(X_test)

# evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error: ', mse)