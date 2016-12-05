# -*- coding: utf-8 -*-

import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def parse(filename):
    """
    Parse file

    :param filename: string
    :return: dataframe
    """

    print '\n Loading... \n'

    dataframe = pd.read_csv(os.path.abspath(os.path.join('app/static/csv/', filename)))

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

    return dataframe, data, target