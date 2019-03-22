"""import librart that will you use"""
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
import pickle
import datetime
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


"""coman-line parsing in the Python standart library"""
parser = argparse.ArgumentParser(description="Train for Boston Housing")
parser.add_argument('--path', type=str, help='Add Datased')
parser.add_argument('--feature_column', type=str, help='Choose Feature for Training')
parser.add_argument('--target_column', type=str, help='Choose Target for Training')
parser.add_argument('--test_size', default=0.3, type=float, help='Give Test Size for Training')
parser.add_argument('--random_state', default= 42, type=int, help='Give Random Size for Training')
parser.add_argument('--algorithm', type=str, help='Choose Algorithm that will you to Use')
parser.add_argument('--save_to', default=None, help='Save the File')
args = parser.parse_args()


"""call file csv in path, so we can call the path from coman-line"""
df = pd.read_csv(args.path, delim_whitespace=True)
"""split the feature and put in variable X"""
col = args.feature_column.split(',')
X = df.loc[:, col]

"""call the target and reshape it"""
y = df[[args.target_column]].values
y = y.reshape(-1,1)
"""split for divide to four variable, and specify test size and random state"""
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = args.test_size,
                                                    random_state= args.random_state)

"""put standar scaler function and fit the fariable and transform it"""
sc_x = StandardScaler()
X_std_train = sc_x.fit_transform(X_train)


"""algorithms for training"""
if args.algorithm == 'rm':
    model = LinearRegression()
    model.fit(X_train, y_train)
elif args.algorithm == 'polynomial':
    polyreg = PolynomialFeatures(degree=2)
    X_train = polyreg.fit_transform(X_train)
    X_test = polyreg.fit_transform(X_test)
    model = LinearRegression()
    model.fit(X_train, y_train)


"""predict X_train and X_test"""
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


"""print the score and accuracy with MSE and r2 score"""
s_train = mean_squared_error(y_train,y_train_pred)
s_test = mean_squared_error(y_test,y_test_pred)
sr2_train = r2_score(y_train, y_train_pred)
sr2_test = r2_score(y_test,y_test_pred)
print(f'Mean Squared error of training set: {s_train}')
print(f'Mean Squared error of testing set: {s_test}')
print(f'R2 variance score of training set: {sr2_train}')
print(f'R2 variance score of testing set: {sr2_test}')


"""save file to folder in pkl_filename that has determined, or you can choose the place of 
folder that will you save the file"""
if args.save_to == None:
    dstr = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    pkl_filename = "weight/boston.sklearn_" + dstr + ".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(forest, file)
else:
    pkl_filename = args.save_to
    with open(pkl_filename, 'wb') as file:
        pkl_filename(forest, file)