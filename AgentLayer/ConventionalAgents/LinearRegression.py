from AgentLayer.ConventionalAgents.Conventional_Models import ConventionalModel
#from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline, convert_daily_return_to_pyfolio_ts
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_score
#import pypfopt
#from pypfopt.efficient_frontier import EfficientFrontier
#from pypfopt import risk_models
import pandas as pd
import pickle
import sys
import traceback
#from pypfopt import EfficientFrontier
#from pypfopt import risk_models
#from pypfopt import expected_returns
#from pypfopt import objective_functions


class LinearRegressionModel(ConventionalModel):

    def __init__(self) -> None:
        super().__init__()
        reg = LinearRegression()
        self.model = reg

    def train_model(self, train_x, train_y):
        '''
        *Trains the model*
        Input: Train data x and train data y
        Output: Linear Regression Model
        '''
        try:
            trained_reg = self.model.fit(train_x, train_y)
            print("Model trained succesfully")
            return trained_reg
        except Exception as e:
            print("ops")

    def predict(self, model, initial_capital, df, unique_trade_date, tech_indicator_list):

        self.initial_capital = initial_capital
        self.df = df
        self.uniqute_trade_date = unique_trade_date
        self.tech_indicator_list = tech_indicator_list

        meta_coefficient = {"date": [], "weights": []}
        self.meta_coefficient = meta_coefficient

        portfolio = pd.DataFrame(index=range(1), columns=unique_trade_date)
        portfolio.loc[0, unique_trade_date[0]] = initial_capital
        self.portfolio = portfolio

        for i in range(len(unique_trade_date) - 1):
            mu, sigma, tics, df_current, df_next = self._return_predict(
                i, self, reference_model=False)
            portfolio_value = self._weight_optimization(
                self, i, mu, sigma, tics, df_current, df_next)

        portfolio = portfolio_value
        portfolio = portfolio.T
        portfolio.columns = ['account_value']
        portfolio = portfolio.reset_index()
        portfolio.columns = ['date', 'account_value']
        stats = backtest_stats(portfolio, value_col_name='account_value')
        portfolio_cumprod = (
            portfolio.account_value.pct_change()+1).cumprod()-1

        return portfolio, stats, portfolio_cumprod, pd.DataFrame(meta_coefficient)

    def _return_predict(i, self, reference_model=False):

        current_date = self.unique_trade_date[i]
        next_date = self.unique_trade_date[i+1]
        df_current = self.df[self.df.date ==
                             current_date].reset_index(drop=True)
        tics = df_current['tic'].values
        features = df_current[self.tech_indicator_list].values
        df_next = self.df[self.df.date == next_date].reset_index(drop=True)
        if not reference_model:
            predicted_y = self.model.predict(features)
            mu = predicted_y
            Sigma = risk_models.sample_cov(
                df_current.return_list[0], returns_data=True)
        else:
            mu = df_next.return_list[0].loc[next_date].values
            Sigma = risk_models.sample_cov(
                df_next.return_list[0], returns_data=True)

        return mu, Sigma, tics, df_current, df_next

    def _weight_optimization(self, i, mu, sigma, tics, df_current, df_next):

        current_date = self.unique_trade_date[i]
        predicted_y_df = pd.DataFrame(
            {"tic": tics.reshape(-1,), "predicted_y": mu.reshape(-1,)})
        min_weight, max_weight = 0, 1

        ef = EfficientFrontier(mu, sigma)
        weights = ef.nonconvex_objective(
            objective_functions.sharpe_ratio,
            objective_args=(ef.expected_returns, ef.cov_matrix),
            weights_sum_to_one=True,
            constraints=[
                # greater than min_weight
                {"type": "ineq", "fun": lambda w: w - min_weight},
                # less than max_weight
                {"type": "ineq", "fun": lambda w: max_weight - w},
            ],
        )

        weight_df = {"tic": [], "weight": []}
        self.meta_coefficient["date"] += [current_date]
        # it = 0
        for item in weights:
            weight_df['tic'] += [item]
            weight_df['weight'] += [weights[item]]

        weight_df = pd.DataFrame(weight_df).merge(predicted_y_df, on=['tic'])
        self.meta_coefficient["weights"] += [weight_df]
        cap = self.portfolio.iloc[0, i]
        # current cash invested for each stock
        current_cash = [element * cap for element in list(weights.values())]
        # current held shares
        current_shares = list(np.array(current_cash) /
                              np.array(df_current.close))
        # next time period price
        next_price = np.array(df_next.close)
        self.portfolio.iloc[0, i+1] = np.dot(current_shares, next_price)

        return self.portfolio

    def save_model(self,model, file_name):
        try:
            with open(file_name, 'wb') as files:
                pickle.dump(model, files)
            print("Model saved succesfully.")
        except (AttributeError,  EOFError, ImportError, IndexError) as e:
            print(traceback.format_exc(e))

    def load_model(self,file_name):
        try:
            with open(file_name, 'rb') as f:
                lr = pickle.load(f)
                print("Model loaded succesfully.")
        except (AttributeError,  EOFError, ImportError, IndexError) as e:
            print(traceback.format_exc(e))

        return lr
