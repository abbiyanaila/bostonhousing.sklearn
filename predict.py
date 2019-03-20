"""import library that will you use"""
import numpy as np
import pandas as pd
import argparse

"""command-line parsing in the Python standard library"""
parser = argparse.ArgumentParser(description='prediction boston housing')
parser.add_argument('--LSTAT', type=float, help='% lower status of the population')
parser.add_argument('--RM', type=float, help='average number of rooms per dwelling')
parser.add_argument('--PTRATIO', type=float, help='pupil-teacher ratio by town')
parser.add_argument('--load', default=None, help='you can load the file from the folder')
args = parser.parse_args()

"""make function predict_boston for declaration features array and load the pkl filename from
the folder"""
def predict_boston(LSTAT, RM, PTRATIO):
    test = np.array([[LSTAT, RM, PTRATIO]])
    if args.load == None:
        clf = pd.read_pickle('weight/boston.sklearn_190319121425.pkl')
        pred = clf.predict(test)
    else:
        clf = args.load
        pred = clf.predict(test)
    return pred[0]

"""build variable which evaluates to the name of current module"""
if __name__ == '__main__':
    """make predict with call the arguments and print it out"""
    out = predict_boston(args.LSTAT, args.RM, args.PTRATIO)
    print(f'The prediction of price is: ${out}')



