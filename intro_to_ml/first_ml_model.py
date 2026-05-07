import pandas as pd

melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data.columns
melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
X.describe()
X.head()

from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

from sklearn.model_selection import train_test_split

# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


#### ====MY TURN====


iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
home_data.columns
y = home_data.SalePrice
iowa_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = home_data[iowa_features]
print(X.describe())
print(X.head())

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)
print("Making predictions for the houses:")
print(X.head())
predictions = iowa_model.predict(X)
print("The predictions are")
print(predictions)



from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))



from sklearn.metrics import mean_absolute_error

mean_absolute_error(val_y, val_predictions)
