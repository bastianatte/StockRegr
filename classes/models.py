import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from utils.misc import get_logger

model_log = get_logger(__name__)
model_log.setLevel(logging.INFO)


class Models(object):
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def random_forest(self):
        """
        Random Forest fit
        :return: prediction
        """
        model = RandomForestRegressor()
        model.fit(self.x_train, self.y_train)
        return model.predict(self.x_test)

    def linear_model(self):
        """
        Linear Regression fit
        :return: prediction
        """
        model = LinearRegression()
        model.fit(self.x_train, self.y_train)
        return model.predict(self.x_test)
