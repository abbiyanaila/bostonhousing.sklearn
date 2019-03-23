# Making Prediction on Boston Housing with Argparse

## Install
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Scikit Learn](https://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install Anaconda distribution of Python, which already has the above packages, and make sure to select Python 3.x Installer.

## Run
In a terminal or command window, navigate to the notebook directory and run one of the following commands:

`iris.ipynb`

This will open the Jupyter Notebook software and project file in your browser.

You can also open the files of python, I use pycharm to bring forward sintaks from `iris.ipynb` to `predic.py` and `train.py`, in those files I use argparse Python standard library, so we can call the parsing argument in the command-line. 

## General Documentation 
To running predic.py :

```html
$ python predict.py --LSTAT 9.1400 --RM 6.4210 --PTRATIO 17.8000
The prediction of price is: $22.61500000000014
```
Here is what is happening :

- Running the script with python and call the file (predic.py) that will we use.
- Call the arguments (--LSTAT, --RM, --PTRATIO) and put content of features, than will show the price of boston housing.

To running `the train.py` :
```html
$ python train.py --path=dataset/housing.data
       CRIM    ZN  INDUS  CHAS  ...   PTRATIO       B  LSTAT  MEDV
0     0.00632  18.0   2.31     0  ...      15.3  396.90   4.98  24.0
1     0.02731   0.0   7.07     0  ...      17.8  396.90   9.14  21.6
2     0.02729   0.0   7.07     0  ...      17.8  392.83   4.03  34.7
3     0.03237   0.0   2.18     0  ...      18.7  394.63   2.94  33.4
4     0.06905   0.0   2.18     0  ...      18.7  396.90   5.33  36.2
```
Running the script with call the argument of --path and choose directory the place of file housing.data saved.

```html
$ python train.py --path=dataset/housing.data --feature_column=LSTAT,RM,PTRATIO
     LSTAT     RM  PTRATIO
0     4.98  6.575     15.3
1     9.14  6.421     17.8
2     4.03  7.185     17.8
3     2.94  6.998     18.7
4     5.33  7.147     18.7
```
Running the script with call argument of --path and --feature_column, --feature_column is the argument for choose the features that will use for training, --feature_column contains are LSTAT, RM, PTRATIO.

```html
$ python train.py --path=dataset/housing.data --feature_column=LSTAT,RM,PTRATIO --target_column=MEDV
[[24. ]
 [21.6]
 [34.7]
 [33.4]
 [36.2]]
```
Running the script with call argument of --path, --feature_column and --target_column, --target_column is the argument for choose the target that will use for training, --target_column contains is MEDV.

```html
$ python train.py --path=dataset/housing.data --feature_column=LSTAT,RM,PTRATIO --target_column=MEDV --algorithm=lr
Mean Squared error of testing set: 27.687244667341563
Mean Squared error of training set: 26.012211922126244
R2 variance score of testing set: 0.6850018441906278
R2 variance score of training set: 0.6509041568614722
```
Runing the script with call argument of --path, --feature_column, --target_column, and --algorithm, --algorithm is the argument contained in algorithms of lr (Linear Regression), polynomial (Polynomial), you can choose algorithm that will you use for training boston housing, and it will show you the Mean Square Error of testing and training and R2 variance testing and training from algorithm that you use.