import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='prediction boston housing')
parser.add_argument('--LSTAT', type=float, help='% lower status of the population')
parser.add_argument('--RM', type=float, help='average number of rooms per dwelling')
parser.add_argument('--PTRATIO', type=float, help='pupil-teacher ratio by town')
parser.add_argument('--load', default=None, help='you can load the file from the folder')
args = parser.parse_args()


def predict_boston(LSTAT, RM, PTRATIO):
    test = np.array([[LSTAT, RM, PTRATIO]])
    if args.load == None:
        clf = pd.read_pickle('weight/boston.sklearn_190319121425.pkl')
        pred = clf.predict(test)
    else:
        clf = args.load
        pred = clf.predict(test)
    return pred[0]

if __name__ == '__main__':
    out = predict_boston(args.LSTAT, args.RM, args.PTRATIO)
    print(f'The prediction of price is: ${out}')



