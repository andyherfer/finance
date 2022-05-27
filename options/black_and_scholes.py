from statistics import NormalDist

import numpy as np
import pandas as pd

from Prices import get_df


def price_option(ticker_df, strike=None, r=0.01988, days=365):
    t = days / 365
    df = ticker_df.copy()
    results = {}
    price_mu = df["Close"].mean()
    range = df["Close"].max() - df["Close"].min()
    sigma = range / price_mu  # la volatilidad del bien subyacente
    S = df["Close"][-1]  # El valor del bien subyacente
    K = S if strike is None else strike  # el precio de ejercicio de la opción
    results["Sigma"] = sigma
    results["Strike"] = K

    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    results["d1"] = d1
    results["d2"] = d2

    Nd1 = NormalDist(mu=0, sigma=1).cdf(d1)
    Nd2 = NormalDist(mu=0, sigma=1).cdf(d2)
    N_minusd2 = NormalDist(mu=0, sigma=1).cdf(-d2)
    N_minusd1 = NormalDist(mu=0, sigma=1).cdf(-d1)

    results["Nd1"] = Nd1
    results["Nd2"] = Nd2

    Call = S * Nd1 - K * np.e ** (-r * t) * Nd2
    Put = K * np.e ** (-r * t) * N_minusd2 - S * N_minusd1

    results["call"] = Call
    results["put"] = Put
    return results


class BlackAndScholes:
    def __init__(
        self, ticker, start_date_for_data="29/03/2021", strike=None, r=0.01988, days=365
    ):
        self.ticker = ticker
        self.start_date_for_data = start_date_for_data
        self.strike = strike
        self.r = r
        self.days = days
        self.ticker_df = get_df(ticker, start_date_for_data)
        self.last_price = self.ticker_df["Close"][-1]

    def price(self, r=None, days=None, strike=None):
        r = self.r if r is None else r
        days = self.days if days is None else days
        strike = self.strike if strike is None else strike
        return price_option(self.ticker_df, strike, r, days)
