import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from utils.misc import get_logger
from config import regress_models_conf as rmc

model_log = get_logger(__name__)
model_log.setLevel(logging.INFO)


class RegressorModels(object):
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def random_forest_regr(self):
        """
        Random Forest fit
        :return: prediction
        """
        model = RandomForestRegressor(max_features=rmc["rf_max_features"])
        model.fit(self.x_train, self.y_train)
        return model.predict(self.x_test)

    def linear_regr(self):
        """
        Linear Regression fit
        :return: prediction
        """
        model = LinearRegression()
        model.fit(self.x_train, self.y_train)
        return model.predict(self.x_test)

    def gradient_boost_regr(self):
        """
        Gradient Boost Regressor fit
        :return: prediction
        """
        model = GradientBoostingRegressor()
        model.fit(self.x_train, self.y_train)
        return model.predict(self.x_test)

    def kneighbors_regr(self):
        """
        K Neighbors regression fit
        :return: prediction
        """
        model = KNeighborsRegressor()
        model.fit(self.x_train, self.y_train)
        return model.predict(self.x_test)

    def lasso_regr(self):
        """
        Lasso Regressor
        :return: prediction
        """
        model = Lasso()
        model.fit(self.x_train, self.y_train)
        return model.predict(self.x_test)

    def elastic_net_regr(self):
        """
        Elastic Net Regressor
        :return: prediction
        """
        model = ElasticNet()
        model.fit(self.x_train, self.y_train)
        return model.predict(self.x_test)

    def decis_tree_regr(self):
        """
        Decision Tree Classifier
        :return: prediction
        """
        model = DecisionTreeRegressor()
        model.fit(self.x_train, self.y_train)
        return model.predict(self.x_test)

    def reg_ensemble(self):
        """
        Regressors Ensemble
        :return: ensempre prediction
        """
        estimators = [
            ("str", DecisionTreeRegressor()),
            ("eln", ElasticNet()),
            ("lasso", Lasso()),
            ("knr", KNeighborsRegressor()),
            ("gbr", GradientBoostingRegressor()),
            ("lr", LinearRegression()),
            ("rf", RandomForestRegressor())

        ]
        reg = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor()
        )
        reg.fit(self.x_train, self.y_train)
        return reg.predict(self.x_test)

    def reg_ensemble_1(self):
        """
        Regressors Ensemble
        :return: ensempre prediction
        """
        estimators = [
            ("dtr", DecisionTreeRegressor()),
            ("eln", ElasticNet()),
            ("lasso", Lasso()),
            ("gbr", GradientBoostingRegressor()),
            ("lr", LinearRegression()),
            ("rf", RandomForestRegressor())

        ]
        reg = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor(n_estimators=10,
                                                                      random_state=42)
        )
        reg.fit(self.x_train, self.y_train)
        return reg.predict(self.x_test)

    def reg_ensemble_2(self):
        """
        Regressors Ensemble
        :return: ensempre prediction
        """
        estimators = [
            ("eln", ElasticNet()),
            ("lasso", Lasso()),
            ("gbr", GradientBoostingRegressor()),
            ("lr", LinearRegression()),
            ("rf", RandomForestRegressor())

        ]
        reg = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor(),
                                cv=5)
        reg.fit(self.x_train, self.y_train)
        return reg.predict(self.x_test)

    def reg_ensemble_3(self):
        """
        Regressors Ensemble
        :return: ensempre prediction
        """
        estimators = [
            ("eln", ElasticNet()),
            ("lasso", Lasso()),
            ("gbr", GradientBoostingRegressor()),
            ("lr", LinearRegression()),
            ("rf", RandomForestRegressor())

        ]
        reg = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor(n_estimators=10,
                                                                      random_state=42)
        )
        reg.fit(self.x_train, self.y_train)
        return reg.predict(self.x_test)