import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle
import datetime
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



parser = argparse.ArgumentParser(description="Train for Boston Housing")
parser.add_argument('--path', type=str, help='Add Datased')
parser.add_argument('--feature_column', type=str, help='Choose Feature for Training')
parser.add_argument('--target_column', type=str, help='Choose Target for Training')
parser.add_argument('--test_size', default=0.3, type=float, help='Give Test Size for Training')
parser.add_argument('--random_state', default= 42, type=int, help='Give Random Size for Training')
parser.add_argument('--algorithm', type=str, help='Choose algorithm that will you to use for training')
args = parser.parse_args()


df = pd.read_csv(args.path, delim_whitespace=True)
col = args.feature_column.split(',')
X = df.loc[:, col]

y = df[[args.target_column]].values
y = y.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = args.test_size,
                                                    random_state= args.random_state)
sc_x = StandardScaler()
X_std_train = sc_x.fit_transform(X_train)

if args.algorithm == 'rm':
    model = LinearRegression()
    model.fit(X_train, y_train)
elif args.algorithm == 'polynomial':
    polyreg = PolynomialFeatures(degree=2)
    X_train = polyreg.fit_transform(X_train)
    X_test = polyreg.fit_transform(X_test)
    model = LinearRegression()
    model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


s_train = mean_squared_error(y_train,y_train_pred)
s_test = mean_squared_error(y_test,y_test_pred)
print(f'Mean Squared error of testing set: {s_train}')
print(f'Mean Squared error of training set: {s_test}')

sr2_train = r2_score(y_train, y_train_pred)
sr2_test = r2_score(y_test,y_test_pred)
print(f'R2 variance score of testing set: {sr2_train}')
print(f'R2 variance score of testing set: {sr2_test}')
