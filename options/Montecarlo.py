import random

from KDEpy import FFTKDE
from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from options_strategy import OptionsStrategy, BSOptionsStrategy

PROFIT_COLOR = "#002856"
LOSS_COLOR = "#f99e74"


class kde:
    def __init__(self, series):
        self.series = series.copy()
        self.kde = None
        self.fitted = False

    def fit(self, *args, **kwargs):
        # Get the standard deviation of the kernel functions
        # Silverman assumes normality of data - use ISJ with much data instead
        self.kernel_std = silvermans_rule(
            self.series.values.reshape(-1, 1)
        )  # Shape (obs, dims)
        return self

    def sample(self, size):
        # (1) First resample original data, then (2) add noise from kernel
        resampled_data = np.random.choice(self.series.values, size=size, replace=True)
        resampled_data = resampled_data + np.random.randn(size) * self.kernel_std
        return resampled_data


class MonteCarlo:
    asset = "asset"

    def __init__(self, series, sim_length=250 * 2, n_sims=10_000):
        price_series = series
        self.series = series
        returns_series = price_series.pct_change().dropna()
        self.n_sims = n_sims
        self.sim_length = sim_length
        self.kde = kde(returns_series).fit()
        self.start_price = price_series.iloc[-1]
        self.sims = None

    def simulate(self):
        """
        The simulate function simulates the price of a stock over time.
        It takes as input a starting price and an expected volatility,
        and returns a list containing the simulated prices using the method described here:

        :param self: Reference the class
        :return: A pandas dataframe with n_sims rows and sim_length columns
        """

        price_series = pd.Series([self.start_price] * self.n_sims)
        samples = self.kde.sample(self.sim_length * self.n_sims)
        samples = pd.DataFrame(samples.reshape(self.sim_length, self.n_sims)) + 1
        samples.loc[-1] = price_series
        samples.sort_index(inplace=True)
        return samples.cumprod()

    def display(self, mc=True, percentiles=True, hist=True):
        """
        The display function displays the results of the Monte Carlo simulation.
        It displays a histogram of the percentiles and prints out a table with
        the mean, standard deviation, median, 5th percentile and 95th percentile.

        :param self: Access the attributes and methods of the class in python
        :return: The plots of the monte carlo simulation and the percentiles
        """

        self.sims = self.simulate()
        if mc:
            self.plot_mc_brute()
        if percentiles:
            self.plot_mc_percentiles()
        if hist:
            self.plot_mc_hist()

    def plot_mc_brute(self, figsize=(12, 10), **kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        for sim in self.sims:
            data_series = self.sims[sim]
            color = self.get_color(data_series)
            if self.n_sims > 200:
                alpha = 1 / np.sqrt(self.n_sims)
            else:
                alpha = 0.6
            ax.plot(data_series, alpha=alpha, color=color, **kwargs)
        ax.hlines(0, data_series.index[0], data_series.index[-1], color="black")
        # Drop Axis lines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        # Change ticks font size
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

        plt.show()

    def get_color(self, series):
        start_price = series.iloc[0]
        end_price = series.iloc[-1]
        color = PROFIT_COLOR if end_price > start_price else LOSS_COLOR
        return color

    def plot_mc_percentiles(self, figsize=(12, 10), **kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        self.sims.T.describe().loc[["75%", "50%", "25%"]].T.plot(
            ax=ax, color=[LOSS_COLOR, PROFIT_COLOR, LOSS_COLOR], **kwargs
        )
        ax.hlines(0, self.sims.index[0], self.sims.index[-1], color="black")
        # Drop Axis lines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        # Change ticks font size
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

        plt.show()

    def plot_mc_hist(self, figsize=(12, 10), **kwargs):
        if self.asset == "asset":
            hist = (self.sims / self.sims.iloc[0] - 1).iloc[-1]
        elif self.asset == "options":
            hist = self.sims.iloc[-1]
        fig, ax = plt.subplots(figsize=figsize)
        hist.name = "Profit at Maturity"
        sns.histplot(
            hist,
            ax=ax,
            bins=int(np.sqrt(self.n_sims)),
            kde=True,
            color="#f99e74",
            **kwargs,
        )
        ax.lines[0].set_color(PROFIT_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        # Change ticks font size
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

        plt.show()


class OptionsMC(MonteCarlo):
    asset = "options"

    def __init__(self, series, sim_length=250 * 2, n_sims=1_000, price_sims=None):
        super().__init__(series, sim_length, n_sims)
        self.options_strategy = OptionsStrategy()
        self.max_loss = 0
        self.price_sims = price_sims

    def plot_strategy(self, figsize=(12, 10), **kwargs):
        min_value = self.series.min() * 0.7
        max_value = self.series.max() * 1.3
        self.options_strategy.plot(int(min_value), int(max_value), **kwargs)

    def add(self, *args, **kwargs):
        self.options_strategy.add(*args, **kwargs)
        return self

    def remove(self, *args, **kwargs):
        self.options_strategy.remove(*args, **kwargs)
        return self

    def clear(self):
        self.options_strategy.clear()
        return self

    def simulate(self):
        """
        The simulate function is a wrapper for the superclass simulate function. It
        applies the get_profit method from the OptionsStrategy class to each price in
        the series returned by super().simulate().

        :param self: Access the attributes and methods of the class in python
        :return: A pandas series with the profit for each simulation
        :doc-author: Trelent
        """
        if self.price_sims is None:
            sims = super().simulate()
            self.price_sims = sims
        else:
            sims = self.price_sims
        sims = sims.apply(
            lambda series: series.apply(
                lambda price: self.options_strategy.get_profit(price)
            )
        )
        self.get_max_loss(sims)
        self.get_expected_utility(sims)
        return sims

    def get_expected_utility(self, sims):
        last_period = sims.iloc[-1]
        expected_utility = last_period.mean()
        expected_utility = expected_utility - sims.iloc[0, 0]
        win_percentage = len(last_period[last_period > 0]) / len(last_period)
        self.utility_obtained = expected_utility * win_percentage

    def get_max_loss(self, sims):
        """
        The get_max_loss function returns the maximum loss of the strategy.

        :param self: Access the attributes and methods of the class in python
        :param sims: A pandas series with the profit for each simulation
        :return: The maximum loss of the strategy
        :doc-author: Trelent
        """

        self.max_loss = sims.min().min()
        return self.max_loss

    def get_color(self, series):
        profit = series.iloc[-1]
        if profit > 0:
            return PROFIT_COLOR
        return LOSS_COLOR

    def display(self,strategy=True, **kwargs):
        """
        The display function plots the strategies of each player and the payoff matrix.


        :param self: Access the attributes and methods of the class in python
        :return: The plot of the strategy
        :doc-author: Trelent
        """
        if strategy:
            self.plot_strategy()
        super().display(**kwargs)


class BSOptionsMC(OptionsMC):
    def __init__(
        self,
        ticker,
        ticker_df=None,
        sim_length=250 * 2,
        n_sims=1_000,
        start_date="29/03/2021",
        r=8.84 / 100,
        available_cash=150,
        price_sims=None,
    ):
        options_strategy = BSOptionsStrategy(
            ticker,
            ticker_df=ticker_df,
            days=sim_length,
            start_date_for_data=start_date,
            r=r,
        )
        series = options_strategy.get_price_series()
        self.last_price = series[-1]
        super().__init__(series, sim_length, n_sims, price_sims=price_sims)
        self.options_strategy = options_strategy
        self.available_cash = available_cash

    def is_a_priori_valid(self):
        # Nos alcanzan las primas
        cash_sum_condition = self.options_strategy.premium <= self.available_cash
        valid_options = True
        for option in self.options_strategy.options:
            if option.premium <= 0:
                valid_options = False
        return valid_options and cash_sum_condition

    def is_a_posteriori_valid(self):
        # La perdida maxima es mayor a lo que podemos perder
        return self.max_loss <= self.available_cash

    def utility(self):
        # Expected Return de la estrategia
        return self.utility_obtained

    def get_results(self):
        self.simulate()
        return {
            "utility": self.utility(),
            "is_valid": self.is_a_priori_valid(),
            "is_valid_posteriori": self.is_a_posteriori_valid(),
        }

    def __repr__(self):
        return "MC(" + "(".join(str(self.options_strategy).split("(")[1:])


sides = ["short", "long"]
kinds = ["put", "call"]
conditions = ["vanilla", "down and out", "up and out", "down and in", "up and in"]
barriers = [i / 100 for i in range(70, 140, 10)]
combinations = [2, 3]
max_strategies = 10000
kwargs_dict = {
    "side": sides,
    "kind": kinds,
    "condition": conditions,
    "barrier": barriers,
}


class GridSearch:
    def __init__(
        self,
        ticker,
        kwargs_dict=kwargs_dict,
        combinations_to_search=combinations,
        n_sims=100,
        max_strategies=max_strategies,
    ):
        self.kwargs_dict = kwargs_dict
        self.combinations_to_search = combinations_to_search
        self.combinations_searched = {}
        self.results = {}
        self.ticker = ticker
        strategy = BSOptionsStrategy(ticker)
        self.ticker_df = strategy.ticker_df

        self.combination_to_search_index = 0
        mc = BSOptionsMC(ticker=self.ticker, ticker_df=self.ticker_df, n_sims=n_sims)
        mc.simulate()
        self.price_sims = mc.price_sims
        self.max_strategies = max_strategies

    def _get_random_kwargs(self):
        new_kwargs = {}
        kwargs_id = ""
        for key, list_ in self.kwargs_dict.items():
            item = random.choice(list_)
            new_kwargs[key] = item
            kwargs_id = kwargs_id + str(item)
        return new_kwargs, kwargs_id

    def get_random_kwargs(self, retries=2):
        for i in range(retries):
            option_kwargs = []
            options_id = []
            for i in range(
                self.combinations_to_search[self.combination_to_search_index]
            ):
                new_kwargs, kwargs_id = self._get_random_kwargs()
                option_kwargs.append(new_kwargs)
                options_id.append(kwargs_id)
            options_id = "".join(sorted(options_id))
            if options_id in self.combinations_searched.keys():
                continue
            else:
                return option_kwargs, options_id

        self.combination_to_search_index += 1
        if self.combination_to_search_index >= len(self.combinations_to_search):
            self.combination_to_search_index = 0
            return None, None
        if len(self.combinations_searched) >= self.max_strategies:
            return None, None
        else:
            return self.get_random_kwargs(retries)

    def setup_sims(self):
        new_kwargs, kwargs_id = self.get_random_kwargs()
        while kwargs_id is not None:
            self.combinations_searched[kwargs_id] = BSOptionsMC(
                ticker=self.ticker, ticker_df=self.ticker_df, price_sims=self.price_sims
            )
            for kwargs in new_kwargs:
                self.combinations_searched[kwargs_id].add(**kwargs)

            new_kwargs, kwargs_id = self.get_random_kwargs()

    def search(self):
        self.setup_sims()
        for key, mc in tqdm(self.combinations_searched.items()):
            self.results[key] = mc.get_results()
        self.results = pd.DataFrame(self.results).T
        self.competed = True

    def get_top(self, k=3):
        results = self.results[
            self.results["is_valid"] & (self.results["is_valid_posteriori"])
        ]
        results = results.sort_values("utility", ascending=False)
        results = results.head(k)
        top_k = []
        for i in results.index:
            top_k.append(self.combinations_searched[i])
        return top_k
