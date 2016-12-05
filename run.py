# -*- coding: utf-8 -*-

import os
import pandas as pd

from app.dataset import parse

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

OUTPUT_FILE = open(os.path.abspath('app/log/logfile.txt'), 'w')

def main():
    """
    Start point

    """
    dataframe, data, target = parse('voice.csv')

    for (index, value) in enumerate(METHOD_NAMES):
        print '{0}) {1}'.format(index + 1, value)

    # Min value 1
    # Max value 2
    # Example: 1,2,3,4,5,6,7,8
    usr_raw = raw_input('\nSelect variations of machine learning methods: ')

    if usr_raw == 'exit':
        pass
    else:
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

        print '\nSearching for best model...\n'
        for random_state in range(30):
            x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=random_state)

            output(pattern='Random state - {0}\n'.format(random_state))
            for index, (name, function) in enumerate(lmethods.iteritems()):
                function.fit(x_train, y_train)
                score = function.score(x_test, y_test)

                if score > best['score']:
                    best['name'] = name
                    best['score'] = score
                    best['random_state'] = random_state

                scores[name].append(score)

                # Write into file
                output(name, score)

            # New row
            # Just for readability
            output(pattern='\n')

        # Best model
        output(pattern='Best model is {0} with score {1} and random state {2}'.format(best['name'], best['score'], best['random_state']))

        # Close file
        OUTPUT_FILE.close()

def output(name=None, score=None, pattern=None):
    """
    Write output into file

    :param name: string
    :param score: float
    :param pattern: string
    """

    if pattern:
        OUTPUT_FILE.write(pattern)
    else:
        OUTPUT_FILE.write('Method - {0} - Score - {1}\n'.format(name, score))

if __name__ == '__main__':
    main();