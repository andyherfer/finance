from statistics import NormalDist

import numpy as np
import pandas as pd
from regex import R

from Prices import get_df

import sys


class recursionlimit:
    def __init__(self, limit):
        self.limit = limit

    def __enter__(self):
        self.old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)

    def __exit__(self, type, value, tb):
        sys.setrecursionlimit(self.old_limit)


def with_recursionlimit(limit):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with recursionlimit(limit):
                return func(*args, **kwargs)

        return wrapper

    return decorator


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

    @with_recursionlimit(10000)
    def look_for_strike(
        self, r=None, days=None, call=None, put=None, tolerance=0.00001
    ):
        assert (call is not None or put is not None) and not (
            call is None and put is None
        ), "You must pass the call xor put price"
        r = self.r if r is None else r
        days = self.days if days is None else days
        if call:
            return self.search_for_call(call, r, days, tolerance)
        if put:
            return self.search_for_put(put, r, days, tolerance)

    def search_for_price(
        self,
        desired_premium,
        r,
        days,
        tolerance=0.0001,
        min_=None,
        max_=None,
        price_to_search="call",
    ):
        """
        The search_for_price function searches for the strike price that gives the
        desired price. Using Binary Search.


        :param self: Access variables that belongs to the class
        :param call: The call price
        :param r: The risk-free interest rate
        :param days: The time to maturity of the option
        :return: The strike price that gives the desired call price
        """
        if max_ is None:
            max_ = self.last_price * 10
        if min_ is None:
            min_ = self.last_price / 10
        while max_ - min_ > tolerance:
            mid = (max_ + min_) / 2
            price = price_option(self.ticker_df, mid, r, days)
            price = price[price_to_search]
            if price > desired_premium:
                max_ = mid
            else:
                min_ = mid
        return mid

    def search_for_call(self, call, r, days, tolerance=0.1):
        return self.search_for_price(call, r, days, tolerance, price_to_search="call")

    def search_for_put(self, put, r, days, tolerance=0.1):
        return self.search_for_price(put, r, days, tolerance, price_to_search="put")
