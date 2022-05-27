from KDEpy import FFTKDE
from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from options_strategy import OptionsStrategy, BSOptionsStrategy


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
        price_series = pd.Series([self.start_price] * self.n_sims)
        samples = self.kde.sample(self.sim_length * self.n_sims)
        samples = pd.DataFrame(samples.reshape(self.sim_length, self.n_sims)) + 1
        samples.loc[-1] = price_series
        samples.sort_index(inplace=True)
        return samples.cumprod()

    def display(self):
        self.sims = self.simulate()
        self.plot_mc_brute()
        self.plot_mc_percentiles()
        self.plot_mc_hist()

    def plot_mc_brute(self, figsize=(12, 10), **kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        for sim in self.sims:
            data_series = self.sims[sim]
            color = self.get_color(data_series)
            ax.plot(data_series, alpha=1 / np.sqrt(self.n_sims), color=color, **kwargs)
        ax.hlines(0, data_series.index[0], data_series.index[-1], color="black")
        plt.show()

    def get_color(self, series):
        start_price = series.iloc[0]
        end_price = series.iloc[-1]
        color = "green" if end_price > start_price else "red"
        return color

    def plot_mc_percentiles(self, figsize=(12, 10), **kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        self.sims.T.describe().loc[["25%", "50%", "75%"]].T.plot(ax=ax, **kwargs)
        ax.hlines(0, self.sims.index[0], self.sims.index[-1], color="black")
        plt.show()

    def plot_mc_hist(self, figsize=(12, 10), **kwargs):
        if self.asset == "asset":
            hist = (self.sims / self.sims.iloc[0] - 1).iloc[-1]
        elif self.asset == "options":
            hist = self.sims.iloc[-1]
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(hist, ax=ax, bins=int(np.sqrt(self.n_sims)), kde=True, **kwargs)
        plt.show()


class OptionsMC(MonteCarlo):
    asset = "options"

    def __init__(self, series, sim_length=250 * 2, n_sims=1_000):
        super().__init__(series, sim_length, n_sims)
        self.options_strategy = OptionsStrategy()

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

    def simulate(self):
        sims = super().simulate()
        sims = sims.apply(
            lambda series: series.apply(
                lambda price: self.options_strategy.get_profit(price)
            )
        )
        return sims

    def get_color(self, series):
        profit = series.iloc[-1]
        if profit > 0:
            return "green"
        return "red"

    def display(self):
        self.plot_strategy()
        super().display()


class BSOptionsMC(OptionsMC):
    def __init__(self, ticker, sim_length=250 * 2, n_sims=1_000):
        options_strategy = BSOptionsStrategy(ticker, days=sim_length)
        series = options_strategy.get_price_series()
        super().__init__(series, sim_length, n_sims)
        self.options_strategy = options_strategy
