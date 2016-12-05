# -*- coding: utf-8 -*-

import os

from app.core import parse
from app.core import logger
from app.core import transform
from app.core import crossValidation

MENU = ['Correlational analysis',
        'Data transformation',
        'Data visualization',
        'Cross validation techniques',
        'Show the scatter plot and histogram charts for all features',]

def main():
    """
    Start point

    """

    dataframe, data, target = parse('voice.csv')

    # Transform status
    STATUS = False

    while True:
        for (index, value) in enumerate(MENU):
            print '{0}) {1}'.format(index + 1, value)

        usr_raw = int(raw_input('Select: '))

        if usr_raw == 1:
            log = logger('logger.correlation.txt')

            log.write('{0}\n {1}'.format(STATUS, data.corr(method='pearson')))
            log.close()

            print '\n--> Done. Results written in logger.correlation.txt\n'
        elif usr_raw == 2:
            # Transform data
            data = transform(data)

            # Change status
            STATUS = True

            print '\n--> Data has been normalized\n'
        elif usr_raw == 3:
            pass
        elif usr_raw == 4:
            crossValidation(data, target)

            print '\n-->  Done. Results written in logger.crossvalidation.txt\n'
        elif usr_raw == 5:
            pass
        else:
            exit()

if __name__ == '__main__':
    main();