from statistics import NormalDist

import numpy as np
import pandas as pd

from Prices import get_df


def price_option(ticker_df, strike=None, r=0.01988, days=365):
    """
    The price_option function computes the price of a call or put option given
    the following parameters:
        - ticker_df: A pandas DataFrame containing the daily stock prices for a specific ticker.
        - strike (optional): The strike price of the option. If not provided, it will be set to S, where S is today's closing price.
        - r (optional): The risk-free interest rate to discount at; defaults to 0.01988 (2% annual).

         Returns a dictionary with two keys: &quot;call&quot; and &quot;put&quot;, each corresponding to their respective prices.

    :param ticker_df: Get the dataframe of the ticker
    :param strike=None: Indicate that the strike price is equal to the last closing price of the stock
    :param r=0.01988: Calculate the discount factor for the future value of money
    :param days=365: Calculate the time to maturity of the option
    :return: A dictionary with the results of the calculation
    """

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
        self.price_series = self.ticker_df["Close"]

    def price(self, r=None, days=None, strike=None):
        """
        The price function computes the price of a European call option given
        the risk-free interest rate, days to expiration, and strike price.


        :param self: Access variables that belongs to the class
        :param r=None: Pass the interest rate used in the price function
        :param days=None: Specify that the days parameter is optional
        :param strike=None: Pass the strike price to the function
        :return: The price of the option
        """

        r = self.r if r is None else r
        days = self.days if days is None else days
        strike = self.strike if strike is None else strike
        return price_option(self.ticker_df, strike, r, days)
