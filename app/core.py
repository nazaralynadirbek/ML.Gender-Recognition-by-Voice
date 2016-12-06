# -*- coding: utf-8 -*-

import os
import operator
import pandas as pd
import seaborn as snb
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso

METHOD_NAMES = ['KNeighborsClassifier', 'DecisionTreeClassifier',
                'GaussianNB', 'LinearRegression', 'LogisticRegression',
                'Ridge', 'Lasso', 'MLPRegressor']

METHOD_FUNCTIONS = [KNeighborsClassifier(), DecisionTreeClassifier(), GaussianNB(),
                    LinearRegression(), LogisticRegression(), Ridge(), Lasso(), MLPRegressor()]

def parse(filename):
    """
    Parse file

    :param filename: string
    :return: dataframe
    """

    print('\n Loading... \n')

    dataframe = pd.read_csv(os.path.abspath(os.path.join('app/static/csv/', filename)), delimiter=',')

    # Classes
    target = dataframe['label']

    le = LabelEncoder()
    le.fit(['male', 'female'])

    # Categorization
    target = le.transform(target)
    
    # Remove from dataframe
    dataframe.drop(['label'], inplace=True, axis=1)

    # Data
    data = pd.get_dummies(dataframe)
    dataframe['label'] = target
    
    return dataframe, data, target

def transform(df):
    """
    Data tranformation

    :param data: dataframe
    :return: dataframe
    """

    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return result

def logger(filename):
    """
    Logger

    :param filename: string
    :return: file
    """
    log = open(os.path.abspath(os.path.join('app/log/', filename)), 'w')

    return log

def crossValidation(data, target):
    """
    Best model

    :param data: dataframe
    :param target: array
    """

    log = logger('logger.crossvalidation.txt')

    for (index, value) in enumerate(METHOD_NAMES):
        print('{0}) {1}'.format(index + 1, value))

    # Min value 1
    # Max value 2
    # Example: 1,2,3,4,5,6,7,8
    usr_raw = input('\nSelect variations of machine learning methods: ')

    # Convert usr_raw to array of integers
    usr_choice = [int(x) for x in usr_raw.split(',')]

    # Selected methods
    lmethods = {}
    for index in usr_choice:
        lmethods[METHOD_NAMES[index - 1]] = METHOD_FUNCTIONS[index - 1]

    # Scores
    scores = dict((key, []) for key in lmethods)

    # Best model
    best = {'name': None,
            'score': 0,
            'random_state': 0}

    print('\nSearching for best model...\n')
    for random_state in range(30):
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=random_state)

        log.write('Random state - {0}\n'.format(random_state))
        for index, (name, function) in enumerate(lmethods.items()):
            function.fit(x_train, y_train)
            score = function.score(x_test, y_test)

            if score > best['score']:
                best['name'] = name
                best['score'] = score
                best['random_state'] = random_state

            scores[name].append(score)

            # Write into file
            log.write('Method - {0} - Score - {1}\n'.format(name, score))

        # New row
        # Just for readability
        log.write('\n')

    # Best model
    log.write('Best model is {0} with score {1} and random state {2}'.format(best['name'], best['score'], best['random_state']))

    # Close file
    log.close()

    # Show accuracy changes
    for (name, value) in scores.items():
        line = plt.plot(range(0, 30), value, 'o-', linewidth=1)
        plt.title('Accuracy of {0}'.format(name))
        plt.show()

def visualization(dataframe, data, target):
    """
    Data visualization

    :param dataframe: dataframe
    :param data: dataframe
    :param target: array
    """

    log = logger('logger.visualization.txt')

    # Write info about dataset
    log.write('Dataset Info\n')
    dataframe.info(buf=log)
    log.write('Shape: {0}'.format(dataframe.shape))

    print('1) Scatter plot')
    print('2) Histogram')
    print('3) Correlation of features with each other')
    print('4) Correlation of features with Target')

    usr_raw = int(input('Select: '))

    if usr_raw == 1:
        pass
    elif usr_raw == 2:
        dataframe.hist()
        plt.show()
    elif usr_raw == 3:
        snb.set(context="paper")
        corr = data.corr()
        f, ax = plt.subplots(figsize=(15, 15))
        plt.title('Correlation matrix for features')
        snb.heatmap(corr, vmax=.8, square=True)
        snb.plt.show()
    elif usr_raw == 4:
        corr = {}
        corr_value = []
        corr_name = []
        for f in data.columns:
            corr[f] = pearsonr(data[f], target)[0]
        for key in corr:
            corr_value.append(corr[key])
            corr_name.append(key)
        snb.barplot(corr_value, corr_name,  capsize=5, orient='h')  
        snb.plt.show()
    else:
        exit()

    # Close
    log.close()