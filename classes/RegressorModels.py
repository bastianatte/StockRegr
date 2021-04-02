import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, LogisticRegression # RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.svm import LinearSVR, SVR
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from utils.misc import get_logger
from config import regress_models_conf as rmc
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict, GridSearchCV
from mlxtend.regressor import StackingRegressor as StackingRegresorMLX
from itertools import combinations

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
        model = RandomForestRegressor(random_state=123)
        model.fit(self.x_train, self.y_train)
        return model, model.predict(self.x_test)

    def linear_regr(self):
        """
        Linear Regression fit
        :return: prediction
        """
        model = LinearRegression()
        model.fit(self.x_train, self.y_train)
        return model, model.predict(self.x_test)

    def gradient_boost_regr(self):
        """
        Gradient Boost Regressor fit
        :return: prediction
        """
        model = GradientBoostingRegressor()
        model.fit(self.x_train, self.y_train)
        return model, model.predict(self.x_test)

    def kneighbors_regr(self):
        """
        K Neighbors regression fit
        :return: prediction
        """
        model = KNeighborsRegressor()
        model.fit(self.x_train, self.y_train)
        return model, model.predict(self.x_test)

    def lasso_regr(self):
        """
        Lasso Regressor
        :return: prediction
        """
        model = Lasso()
        model.fit(self.x_train, self.y_train)
        return model, model.predict(self.x_test)

    def elastic_net_regr(self):
        """
        Elastic Net Regressor
        :return: prediction
        """
        model = ElasticNet()
        model.fit(self.x_train, self.y_train)
        return model, model.predict(self.x_test)

    def decis_tree_regr(self):
        """
        Decision Tree Classifier
        :return: prediction
        """
        model = DecisionTreeRegressor()
        model.fit(self.x_train, self.y_train)
        return model, model.predict(self.x_test)

    def voting_regressor_ensemble_1(self):
        lr, lr_pred = self.linear_regr()
        lasso, lasso_pred = self.lasso_regr()
        rf, rf_pred = self.random_forest_regr()
        er = VotingRegressor([
            ('lr', lr),
            ('lasso', lasso),
            ("rf", rf)
            ], n_jobs=-1)
        return er.fit(self.x_train, self.y_train).predict(self.x_test)

    def voting_regressor_ensemble_2(self):
        lr, lr_pred = self.linear_regr()
        rf, rf_pred = self.random_forest_regr()
        er = VotingRegressor([
            ('lr', lr),
            ('rf', rf),
            ], n_jobs=-1)
        return er.fit(self.x_train, self.y_train).predict(self.x_test)

    def voting_regressor_ensemble_3(self):
        lr, lr_pred = self.linear_regr()
        rf, rf_pred = self.random_forest_regr()
        er = VotingRegressor([
            ('lr', lr),
            ('rf', rf),
            ], n_jobs=-1)
        return er.fit(self.x_train, self.y_train).predict(self.x_test)

    def reg_ensemble_1(self):
        """
        Regressors Ensemble
        :return: ensempre prediction
        """
        lr, lr_pred = self.linear_regr()
        rf, rf_pred = self.random_forest_regr()
        lasso, lasso_pred = self.lasso_regr()
        # el, el_pred = self.elastic_net_regr()
        # dt, dt_pred = self.decis_tree_regr()
        # knr, knr_pred = self.kneighbors_regr()
        # gbr, gbr_pred = self.gradient_boost_regr()
        estimators = [
            # ("str", dt),
            # ("eln", el),
            ("lasso", lasso),
            # ("knr", knr),
            # ("gbr", gbr),
            ("lr", lr),
            ("rf", rf)
        ]
        reg = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor(),
                                n_jobs=-1)
        reg.fit(self.x_train, self.y_train)
        return reg.predict(self.x_test)

    def reg_ensemble_2(self):
        """
        Regressors Ensemble
        :return: ensempre prediction
        """
        lr, lr_pred = self.linear_regr()
        rf, rf_pred = self.random_forest_regr()
        lasso, lasso_pred = self.lasso_regr()
        lor = LogisticRegression()
        # el, el_pred = self.elastic_net_regr()
        estimators = [
            # ("eln", el),
            ("lasso", lasso),
            ("lr", lr),
            ("rf", rf)
        ]
        reg = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor(),
                                cv=5, #10
                                n_jobs=-1)
        reg.fit(self.x_train, self.y_train)
        return reg.predict(self.x_test)

    def reg_ensemble_3(self):
        """
        Regressors Ensemble
        :return: ensempre prediction
        """
        lr, lr_pred = self.linear_regr()
        rf, rf_pred = self.random_forest_regr()
        lasso, lasso_pred = self.lasso_regr()
        # el, el_pred = self.elastic_net_regr()
        estimators = [
            # ("eln", el),
            ("lasso", lasso),
            ("lr", lr),
            ("rf", rf)

        ]
        reg = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor(),
                                cv=50,
                                n_jobs=-1)
        reg.fit(self.x_train, self.y_train)
        return reg.predict(self.x_test)

    def reg_ensemble_4(self):
        """
        Regressors Ensemble
        :return: ensempre prediction
        """
        lr, lr_pred = self.linear_regr()
        rf, rf_pred = self.random_forest_regr()
        lasso, lasso_pred = self.lasso_regr()
        estimators = [
            ("lr", lr),
            ("rf", rf),
            ("lasso", lasso)
        ]
        reg = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor(),
                                cv=200,
                                n_jobs=-1)
        reg.fit(self.x_train, self.y_train)
        return reg.predict(self.x_test)

    def reg_ensemble_5(self):
        """
        Regressors Ensemble
        :return: ensempre prediction
        """
        param = {'final_estimator__max_features': [1, 5],
                 'final_estimator__n_jobs': [1, -1, 5]}
        lr, lr_pred = self.linear_regr()
        rf, rf_pred = self.random_forest_regr()
        estimators = [
            ("lr", lr),
            ("rf", rf)
        ]
        # tss = TimeSeriesSplit(n_splits=2, test_size=10)
        tss = TimeSeriesSplit(gap=20, max_train_size=None, n_splits=10, test_size=None)
        reg = StackingRegressor(estimators=estimators,
                                final_estimator=RandomForestRegressor(),
                                cv=tss,
                                n_jobs=-1)
        reg.fit(self.x_train, self.y_train)
        return reg.predict(self.x_test)

    def mlx_reg_1(self):
        lr, lr_pred = self.linear_regr()
        rf, rf_pred = self.random_forest_regr()
        lasso, lasso_pred = self.lasso_regr()
        sclf = StackingRegresorMLX(
            regressors=[lr, rf, lasso],
            meta_regressor=RandomForestRegressor(ccp_alpha=0.1,
                                                 max_features="auto",
                                                 n_estimators=30)
        )
        sclf.fit(self.x_train, self.y_train)
        return sclf.predict(self.x_test)

    def mlx_reg_2(self):
        lr, lr_pred = self.linear_regr()
        rf, rf_pred = self.random_forest_regr()
        lasso, lasso_pred = self.lasso_regr()
        # lor = LogisticRegression()
        sclf = StackingRegresorMLX(
            regressors=[lr, rf, lasso],
            meta_regressor=RandomForestRegressor(max_features="log2",
                                                 n_estimators=500,
                                                 n_jobs=-1))

        params = {'lasso__alpha': [1.0, 10.0, 0.1],
                  'meta_regressor__n_estimators': [10, 50, 100],
                  'meta_regressor__ccp_alpha': np.arange(0, 1, 0.001).tolist(),
                  "meta_regressor__max_features": ["auto", "sqrt", "log2"],
                  }

        grid = GridSearchCV(
            estimator=sclf,
            param_grid=params,
            cv=3,
            refit=True
        )
        # print(sclf.get_params().keys())
        grid.fit(self.x_train, self.y_train)
        print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
        return grid.predict(self.x_test)

    def ensemble_loop(self):
        model_dict = {
            "lr": LinearRegression(),
            "rf": RandomForestRegressor(),
            # "lasso": Lasso(),
            # "el": ElasticNet(),
            "dt": DecisionTreeRegressor(),
            "kn": KNeighborsRegressor(),
            "bgr": GradientBoostingRegressor()
        }
        # for ncomb in range(2, 4):
        comb = combinations(model_dict, 3)
        for i in list(comb):
            models = [model_dict[y] for y in i]
            # print(i, models)
            label = "_".join(i) + "_ens"
            yield label, StackingRegresorMLX(regressors=models, meta_regressor=RandomForestRegressor(random_state=123,
                                                                                                     n_jobs=-1))

    # def fitpred_ensemble(self):
    #     for idx, mod in self.ensemble_loop():
    #         mod.fit(self.x_train, self.y_train)
    #         yield idx, mod.predict(self.x_test)

    def fitpred_ensemble(self):
        for idx, mod in self.ensemble_loop():
            params = {'meta_regressor__n_estimators': [500, 250, 700],
                      'meta_regressor__min_samples_leaf': [2, 5, 8],
                      "meta_regressor__max_features": [1, "sqrt", 0.1]
                      }

            grid = GridSearchCV(
                estimator=mod,
                param_grid=params,
                cv=3,
                refit=True
            )
            grid.fit(self.x_train, self.y_train)
            print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
            yield idx, grid.predict(self.x_test)
