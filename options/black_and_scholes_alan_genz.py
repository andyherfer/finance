# %%

from statistics import NormalDist
from scipy.stats.distributions import chi2
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


# %%
def Left(string, n):
    return string[:n]


def Right(string, n):
    return string[-n:]


def CND(value, mu=0, sigma=1):
    return NormalDist(mu=mu, sigma=sigma).cdf(value)


def ESoftBarrier(
    OutPutFlag: str,
    TypeFlag: str,
    S: float,
    X: float,
    L: float,
    U: float,
    T: float,
    r: float,
    b: float,
    v: float,
    dS=None,
):
    if dS is None:
        dS = 0.0001

    if OutPutFlag == "p":  # Value
        ESoftBarrier = SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v)
    elif OutPutFlag == "d":  # Delta
        ESoftBarrier = (
            SoftBarrier(TypeFlag, S + dS, X, L, U, T, r, b, v)
            - SoftBarrier(TypeFlag, S - dS, X, L, U, T, r, b, v)
        ) / (2 * dS)
    elif OutPutFlag == "dddv":  # DeltaDVol
        ESoftBarrier = (
            1
            / (4 * dS * 0.01)
            * (
                SoftBarrier(TypeFlag, S + dS, X, L, U, T, r, b, v + 0.01)
                - SoftBarrier(TypeFlag, S + dS, X, L, U, T, r, b, v - 0.01)
                - SoftBarrier(TypeFlag, S - dS, X, L, U, T, r, b, v + 0.01)
                + SoftBarrier(TypeFlag, S - dS, X, L, U, T, r, b, v - 0.01)
            )
            / 100
        )
    elif OutPutFlag == "g":  # Gamma
        ESoftBarrier = (
            SoftBarrier(TypeFlag, S + dS, X, L, U, T, r, b, v)
            - 2 * SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v)
            + SoftBarrier(TypeFlag, S - dS, X, L, U, T, r, b, v)
        ) / (dS ^ 2)
    elif OutPutFlag == "gp":  # GammaP
        ESoftBarrier = (
            S / 100 * ESoftBarrier("g", TypeFlag, S + dS, X, L, U, T, r, b, v)
        )
    elif OutPutFlag == "gv":  # DGammaDVol
        ESoftBarrier = (
            (
                SoftBarrier(TypeFlag, S + dS, X, L, U, T, r, b, v + 0.01)
                - 2 * SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v + 0.01)
                + SoftBarrier(TypeFlag, S - dS, X, L, U, T, r, b, v + 0.01)
                - SoftBarrier(TypeFlag, S + dS, X, L, U, T, r, b, v - 0.01)
                + 2 * SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v - 0.01)
                - SoftBarrier(TypeFlag, S - dS, X, L, U, T, r, b, v - 0.01)
            )
            / (2 * 0.01 * dS ^ 2)
            / 100
        )
    elif OutPutFlag == "v":  # Vega
        ESoftBarrier = (
            SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v + 0.01)
            - SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v - 0.01)
        ) / 2
    elif OutPutFlag == "dvdv":  # DVegaDVol/Vomma
        ESoftBarrier = (
            SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v + 0.01)
            - 2 * SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v)
            + SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v - 0.01)
        ) / 0.01 ^ 2 / 10000
    elif OutPutFlag == "vp":  # VegaP
        ESoftBarrier = (
            v / 0.1 * ESoftBarrier("v", TypeFlag, S + dS, X, L, U, T, r, b, v)
        )
    elif OutPutFlag == "r":  # Rho
        ESoftBarrier = (
            SoftBarrier(TypeFlag, S, X, L, U, T, r + 0.01, b + 0.01, v)
            - SoftBarrier(TypeFlag, S, X, L, U, T, r - 0.01, b - 0.01, v)
        ) / 2
    elif OutPutFlag == "f":  # Rho2/Phi
        ESoftBarrier = (
            SoftBarrier(TypeFlag, S, X, L, U, T, r, b - 0.01, v)
            - SoftBarrier(TypeFlag, S, X, L, U, T, r, b + 0.01, v)
        ) / 2
    elif OutPutFlag == "b":  # Carry sensitivity
        ESoftBarrier = (
            SoftBarrier(TypeFlag, S, X, L, U, T, r, b + 0.01, v)
            - SoftBarrier(TypeFlag, S, X, L, U, T, r, b - 0.01, v)
        ) / 2
    elif OutPutFlag == "t":  # Theta
        if T <= 1 / 365:
            ESoftBarrier = SoftBarrier(
                TypeFlag, S, X, L, U, 1e-05, r, b, v
            ) - SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v)
        else:
            ESoftBarrier = SoftBarrier(
                TypeFlag, S, X, L, U, T - 1 / 365, r, b, v
            ) - SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v)
    elif OutPutFlag == "dx":  # Strike Delta
        ESoftBarrier = (
            SoftBarrier(TypeFlag, S, X + dS, L, U, T, r, b, v)
            - SoftBarrier(TypeFlag, S, X - dS, L, U, T, r, b, v)
        ) / (2 * dS)
    elif OutPutFlag == "dxdx":  # Strike Gamma
        ESoftBarrier == (
            SoftBarrier(TypeFlag, S, X + dS, L, U, T, r, b, v)
            - 2 * SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v)
            + SoftBarrier(TypeFlag, S, X - dS, L, U, T, r, b, v)
        ) / (dS ^ 2)
    return ESoftBarrier


def Exp(value):
    return np.exp(value)


def Log(value):
    return np.log(value)


def Sqr(value):
    return np.sqrt(value)


# Soft barrier options
def SoftBarrier(
    TypeFlag: str,
    S: float,
    X: float,
    L: float,
    U: float,
    T: float,
    r: float,
    b: float,
    v: float,
):
    if TypeFlag in ["cdi", "cdo"]:
        eta = 1
    else:
        eta = -1

    mu = (b + v ^ 2 / 2) / v ^ 2
    Lambda1 = Exp(-1 / 2 * v ^ 2 * T * (mu + 0.5) * (mu - 0.5))
    Lambda2 = Exp(-1 / 2 * v ^ 2 * T * (mu - 0.5) * (mu - 1.5))
    d1 = Log(U ^ 2 / (S * X)) / (v * Sqr(T)) + mu * v * Sqr(T)
    d2 = d1 - (mu + 0.5) * v * Sqr(T)
    d3 = Log(U ^ 2 / (S * X)) / (v * Sqr(T)) + (mu - 1) * v * Sqr(T)
    d4 = d3 - (mu - 0.5) * v * Sqr(T)
    e1 = Log(L ^ 2 / (S * X)) / (v * Sqr(T)) + mu * v * Sqr(T)
    e2 = e1 - (mu + 0.5) * v * Sqr(T)
    e3 = Log(L ^ 2 / (S * X)) / (v * Sqr(T)) + (mu - 1) * v * Sqr(T)
    e4 = e3 - (mu - 0.5) * v * Sqr(T)

    Value = (
        eta
        * 1
        / (U - L)
        * (
            S * Exp((b - r) * T) * S
            ^ (-2 * mu) * (S * X)
            ^ (mu + 0.5)
            / (2 * (mu + 0.5))
            * (
                (U ^ 2 / (S * X))
                ^ (mu + 0.5) * CND(eta * d1)
                - Lambda1 * CND(eta * d2)
                - (L ^ 2 / (S * X))
                ^ (mu + 0.5) * CND(eta * e1) + Lambda1 * CND(eta * e2)
            )
            - X * Exp(-r * T) * S
            ^ (-2 * (mu - 1)) * (S * X)
            ^ (mu - 0.5)
            / (2 * (mu - 0.5))
            * (
                (U ^ 2 / (S * X))
                ^ (mu - 0.5) * CND(eta * d3)
                - Lambda2 * CND(eta * d4)
                - (L ^ 2 / (S * X))
                ^ (mu - 0.5) * CND(eta * e3) + Lambda2 * CND(eta * e4)
            )
        )
    )

    if TypeFlag in ["cdi", "pui"]:
        SoftBarrier = Value
    elif TypeFlag == "cdo":
        SoftBarrier = GBlackScholes("c", S, X, T, r, b, v) - Value
    elif TypeFlag == "puo":
        SoftBarrier = GBlackScholes("p", S, X, T, r, b, v) - Value
    return SoftBarrier


def ArcSin(value):
    return np.arcsin(value)


def ELookBarrier(
    OutPutFlag: str,
    TypeFlag: str,
    S: float,
    X: float,
    H: float,
    t1: float,
    T2: float,
    r: float,
    b: float,
    v: float,
    dS=None,
):

    if dS is None:
        dS = 0.0001

    CallPutFlag = Left(TypeFlag, 1)

    if (TypeFlag == "cuo" and S >= H) or (TypeFlag == "pdo" and S <= H):
        ELookBarrier = 0
    elif (TypeFlag == "cui" and S >= H) or (TypeFlag == "pdi" and S <= H):
        ELookBarrier = PartialFixedLB(CallPutFlag, S, X, t1, T2, r, b, v)
        return ELookBarrier

    if OutPutFlag == "p":  # Value
        ELookBarrier = LookBarrier(TypeFlag, S, X, H, t1, T2, r, b, v)
    elif OutPutFlag == "d":  # Delta
        ELookBarrier = (
            LookBarrier(TypeFlag, S + dS, X, H, t1, T2, r, b, v)
            - LookBarrier(TypeFlag, S - dS, X, H, t1, T2, r, b, v)
        ) / (2 * dS)
    elif OutPutFlag == "dddv":  # DeltaDVol
        ELookBarrier = (
            1
            / (4 * dS * 0.01)
            * (
                LookBarrier(TypeFlag, S + dS, X, H, t1, T2, r, b, v + 0.01)
                - LookBarrier(TypeFlag, S + dS, X, H, t1, T2, r, b, v - 0.01)
                - LookBarrier(TypeFlag, S - dS, X, H, t1, T2, r, b, v + 0.01)
                + LookBarrier(TypeFlag, S - dS, X, H, t1, T2, r, b, v - 0.01)
            )
            / 100
        )
    elif OutPutFlag == "g":  # Gamma
        ELookBarrier = (
            LookBarrier(TypeFlag, S + dS, X, H, t1, T2, r, b, v)
            - 2 * LookBarrier(TypeFlag, S, X, H, t1, T2, r, b, v)
            + LookBarrier(TypeFlag, S - dS, X, H, t1, T2, r, b, v)
        ) / (dS ^ 2)

    elif OutPutFlag == "gp":  # GammaP
        ELookBarrier = (
            S / 100 * ELookBarrier("g", TypeFlag, S + dS, X, H, t1, T2, r, b, v)
        )

    elif OutPutFlag == "gv":  # DGammaDvol
        ELookBarrier = (
            (
                LookBarrier(TypeFlag, S + dS, X, H, t1, T2, r, b, v + 0.01)
                - 2 * LookBarrier(TypeFlag, S, X, H, t1, T2, r, b, v + 0.01)
                + LookBarrier(TypeFlag, S - dS, X, H, t1, T2, r, b, v + 0.01)
                - LookBarrier(TypeFlag, S + dS, X, H, t1, T2, r, b, v - 0.01)
                + 2 * LookBarrier(TypeFlag, S, X, H, t1, T2, r, b, v - 0.01)
                - LookBarrier(TypeFlag, S - dS, X, H, t1, T2, r, b, v - 0.01)
            )
            / (2 * 0.01 * dS ^ 2)
            / 100
        )
    elif OutPutFlag == "v":  # Vega
        ELookBarrier = (
            LookBarrier(TypeFlag, S, X, H, t1, T2, r, b, v + 0.01)
            - LookBarrier(TypeFlag, S, X, H, t1, T2, r, b, v - 0.01)
        ) / 2

    elif OutPutFlag == "vp":  # VegaP
        ELookBarrier = (
            v / 0.1 * ELookBarrier("v", TypeFlag, S + dS, X, H, t1, T2, r, b, v)
        )

    elif OutPutFlag == "dvdv":  # DvegaDvol/vomma
        ELookBarrier = (
            LookBarrier(TypeFlag, S, X, H, t1, T2, r, b, v + 0.01)
            - 2 * LookBarrier(TypeFlag, S, X, H, t1, T2, r, b, v)
            + LookBarrier(TypeFlag, S, X, H, t1, T2, r, b, v - 0.01)
        ) / 0.01 ^ 2 / 10000
    elif OutPutFlag == "r":  # Rho
        ELookBarrier = (
            LookBarrier(TypeFlag, S, X, H, t1, T2, r + 0.01, b + 0.01, v)
            - LookBarrier(TypeFlag, S, X, H, t1, T2, r - 0.01, b - 0.01, v)
        ) / 2
    elif OutPutFlag == "f":  # Rho2 Phi
        ELookBarrier = (
            LookBarrier(TypeFlag, S, X, H, t1, T2, r, b - 0.01, v)
            - LookBarrier(TypeFlag, S, X, H, t1, T2, r, b + 0.01, v)
        ) / 2
    elif OutPutFlag == "b":  # Carry sensitivity
        ELookBarrier = (
            LookBarrier(TypeFlag, S, X, H, t1, T2, r, b + 0.01, v)
            - LookBarrier(TypeFlag, S, X, H, t1, T2, r, b - 0.01, v)
        ) / 2
    elif OutPutFlag == "t":  # Theta
        if t1 <= 1 / 365:
            ELookBarrier = LookBarrier(
                TypeFlag, S, X, H, 1e-05, T2 - 1 / 365, r, b, v
            ) - LookBarrier(TypeFlag, S, X, H, t1, T2, r, b, v)
        else:
            ELookBarrier = LookBarrier(
                TypeFlag, S, X, H, t1 - 1 / 365, T2 - 1 / 365, r, b, v
            ) - LookBarrier(TypeFlag, S, X, H, t1, T2, r, b, v)

    elif OutPutFlag == "dx":  # Strike Delta
        ELookBarrier = (
            LookBarrier(TypeFlag, S, X + dS, H, t1, T2, r, b, v)
            - LookBarrier(TypeFlag, S, X - dS, H, t1, T2, r, b, v)
        ) / (2 * dS)
    elif OutPutFlag == "dxdx":  # Strike Gamma
        ELookBarrier = (
            LookBarrier(TypeFlag, S, X + dS, H, t1, T2, r, b, v)
            - 2 * LookBarrier(TypeFlag, S, X, H, t1, T2, r, b, v)
            + LookBarrier(TypeFlag, S, X - dS, H, t1, T2, r, b, v)
        ) / (dS ^ 2)

    return ELookBarrier


def Min(*args):
    return np.min(args)


def Max(*args):
    return np.max(args)


# // Look-barrier options
def LookBarrier(
    TypeFlag: str,
    S: float,
    X: float,
    H: float,
    t1: float,
    T2: float,
    r: float,
    b: float,
    v: float,
):
    hh = Log(H / S)
    k = Log(X / S)
    mu1 = b - v ^ 2 / 2
    mu2 = b + v ^ 2 / 2
    rho = Sqr(t1 / T2)

    if TypeFlag in ["cuo", "cui"]:
        eta = 1
        m = Min(hh, k)
    elif TypeFlag in ["pdo", "pdi"]:
        eta = -1
        m = Max(hh, k)

    g1 = (
        CND(eta * (hh - mu2 * t1) / (v * Sqr(t1)))
        - Exp(2 * mu2 * hh / v ^ 2) * CND(eta * (-hh - mu2 * t1) / (v * Sqr(t1)))
    ) - (
        CND(eta * (m - mu2 * t1) / (v * Sqr(t1)))
        - Exp(2 * mu2 * hh / v ^ 2) * CND(eta * (m - 2 * hh - mu2 * t1) / (v * Sqr(t1)))
    )
    g2 = (
        CND(eta * (hh - mu1 * t1) / (v * Sqr(t1)))
        - Exp(2 * mu1 * hh / v ^ 2) * CND(eta * (-hh - mu1 * t1) / (v * Sqr(t1)))
    ) - (
        CND(eta * (m - mu1 * t1) / (v * Sqr(t1)))
        - Exp(2 * mu1 * hh / v ^ 2) * CND(eta * (m - 2 * hh - mu1 * t1) / (v * Sqr(t1)))
    )
    part1 = (
        S
        * Exp((b - r) * T2)
        * (1 + v ^ 2 / (2 * b))
        * (
            CBND(
                eta * (m - mu2 * t1) / (v * Sqr(t1)),
                eta * (-k + mu2 * T2) / (v * Sqr(T2)),
                -rho,
            )
            - Exp(2 * mu2 * hh / v ^ 2)
            * CBND(
                eta * (m - 2 * hh - mu2 * t1) / (v * Sqr(t1)),
                eta * (2 * hh - k + mu2 * T2) / (v * Sqr(T2)),
                -rho,
            )
        )
    )
    part2 = (
        -Exp(-r * T2)
        * X
        * (
            CBND(
                eta * (m - mu1 * t1) / (v * Sqr(t1)),
                eta * (-k + mu1 * T2) / (v * Sqr(T2)),
                -rho,
            )
            - Exp(2 * mu1 * hh / v ^ 2)
            * CBND(
                eta * (m - 2 * hh - mu1 * t1) / (v * Sqr(t1)),
                eta * (2 * hh - k + mu1 * T2) / (v * Sqr(T2)),
                -rho,
            )
        )
    )
    part3 = -Exp(-r * T2) * v ^ 2 / (2 * b) * (
        S * (S / X)
        ^ (-2 * b / v ^ 2)
        * CBND(
            eta * (m + mu1 * t1) / (v * Sqr(t1)),
            eta * (-k - mu1 * T2) / (v * Sqr(T2)),
            -rho,
        )
        - H * (H / X)
        ^ (-2 * b / v ^ 2)
        * CBND(
            eta * (m - 2 * hh + mu1 * t1) / (v * Sqr(t1)),
            eta * (2 * hh - k - mu1 * T2) / (v * Sqr(T2)),
            -rho,
        )
    )
    part4 = (
        S
        * Exp((b - r) * T2)
        * (
            (1 + v ^ 2 / (2 * b)) * CND(eta * mu2 * (T2 - t1) / (v * Sqr(T2 - t1)))
            + Exp(-b * (T2 - t1))
            * (1 - v ^ 2 / (2 * b))
            * CND(eta * (-mu1 * (T2 - t1)) / (v * Sqr(T2 - t1)))
        )
        * g1
        - Exp(-r * T2) * X * g2
    )
    OutValue = eta * (part1 + part2 + part3 + part4)

    if TypeFlag in ["cuo", "pdo"]:
        LookBarrier = OutValue
    elif TypeFlag == "cui":
        LookBarrier = PartialFixedLB("c", S, X, t1, T2, r, b, v) - OutValue
    elif TypeFlag == "pdi":
        LookBarrier = PartialFixedLB("p", S, X, t1, T2, r, b, v) - OutValue

    return LookBarrier


def EPartialTimeBarrier(
    OutPutFlag: str,
    TypeFlag: str,
    S: float,
    X: float,
    H: float,
    t1: float,
    T2: float,
    r: float,
    b: float,
    v: float,
    dS=None,
):

    if dS is None:
        dS = 0.0001

    if (
        (TypeFlag == "cuoA" and S >= H)
        or (TypeFlag == "puoA" and S >= H)
        or (TypeFlag == "cdoA" and S <= H)
        or (TypeFlag == "pdoA" and S <= H)
    ):
        EPartialTimeBarrier = 0
        return EPartialTimeBarrier

    if OutPutFlag == "p":  # Value
        EPartialTimeBarrier = PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b, v)
    elif OutPutFlag == "d":  # Delta
        EPartialTimeBarrier = (
            PartialTimeBarrier(TypeFlag, S + dS, X, H, t1, T2, r, b, v)
            - PartialTimeBarrier(TypeFlag, S - dS, X, H, t1, T2, r, b, v)
        ) / (2 * dS)
    elif OutPutFlag == "dddv":  # DeltaDVol
        EPartialTimeBarrier = (
            1
            / (4 * dS * 0.01)
            * (
                PartialTimeBarrier(TypeFlag, S + dS, X, H, t1, T2, r, b, v + 0.01)
                - PartialTimeBarrier(TypeFlag, S + dS, X, H, t1, T2, r, b, v - 0.01)
                - PartialTimeBarrier(TypeFlag, S - dS, X, H, t1, T2, r, b, v + 0.01)
                + PartialTimeBarrier(TypeFlag, S - dS, X, H, t1, T2, r, b, v - 0.01)
            )
            / 100
        )
    elif OutPutFlag == "g":  # Gamma
        EPartialTimeBarrier = (
            PartialTimeBarrier(TypeFlag, S + dS, X, H, t1, T2, r, b, v)
            - 2 * PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b, v)
            + PartialTimeBarrier(TypeFlag, S - dS, X, H, t1, T2, r, b, v)
        ) / (dS ^ 2)
    elif OutPutFlag == "gp":  # GammaP
        EPartialTimeBarrier = (
            S / 100 * EPartialTimeBarrier("g", TypeFlag, S + dS, X, H, t1, T2, r, b, v)
        )
    elif OutPutFlag == "gv":  # DGammaDvol
        EPartialTimeBarrier = (
            (
                PartialTimeBarrier(TypeFlag, S + dS, X, H, t1, T2, r, b, v + 0.01)
                - 2 * PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b, v + 0.01)
                + PartialTimeBarrier(TypeFlag, S - dS, X, H, t1, T2, r, b, v + 0.01)
                - PartialTimeBarrier(TypeFlag, S + dS, X, H, t1, T2, r, b, v - 0.01)
                + 2 * PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b, v - 0.01)
                - PartialTimeBarrier(TypeFlag, S - dS, X, H, t1, T2, r, b, v - 0.01)
            )
            / (2 * 0.01 * dS ^ 2)
            / 100
        )
    elif OutPutFlag == "v":  # Vega
        EPartialTimeBarrier = (
            PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b, v + 0.01)
            - PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b, v - 0.01)
        ) / 2
    elif OutPutFlag == "vp":  # VegaP
        EPartialTimeBarrier = (
            v / 0.1 * EPartialTimeBarrier("v", TypeFlag, S + dS, X, H, t1, T2, r, b, v)
        )
    elif OutPutFlag == "dvdv":  # DvegaDvol/vomma
        EPartialTimeBarrier = (
            PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b, v + 0.01)
            - 2 * PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b, v)
            + PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b, v - 0.01)
        ) / 0.01 ^ 2 / 10000
    elif OutPutFlag == "r":  # Rho
        EPartialTimeBarrier = (
            PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r + 0.01, b + 0.01, v)
            - PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r - 0.01, b - 0.01, v)
        ) / 2
    elif OutPutFlag == "fr":  # Futures option Rho
        EPartialTimeBarrier = (
            PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r + 0.01, 0, v)
            - PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r - 0.01, 0, v)
        ) / 2
    elif OutPutFlag == "f":  # Rho2 Phi
        EPartialTimeBarrier = (
            PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b - 0.01, v)
            - PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b + 0.01, v)
        ) / 2
    elif OutPutFlag == "b":  # Carry sensitivity
        EPartialTimeBarrier = (
            PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b + 0.01, v)
            - PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b - 0.01, v)
        ) / 2
    elif OutPutFlag == "t":  # Theta
        if t1 <= 1 / 365:
            EPartialTimeBarrier = PartialTimeBarrier(
                TypeFlag, S, X, H, 1e-05, T2 - 1 / 365, r, b, v
            ) - PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b, v)
        else:
            EPartialTimeBarrier = PartialTimeBarrier(
                TypeFlag, S, X, H, t1 - 1 / 365, T2 - 1 / 365, r, b, v
            ) - PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b, v)

    elif OutPutFlag == "dx":  # Strike Delta
        EPartialTimeBarrier = (
            PartialTimeBarrier(TypeFlag, S, X + dS, H, t1, T2, r, b, v)
            - PartialTimeBarrier(TypeFlag, S, X - dS, H, t1, T2, r, b, v)
        ) / (2 * dS)
    elif OutPutFlag == "dxdx":  # Strike Gamma
        EPartialTimeBarrier = (
            PartialTimeBarrier(TypeFlag, S, X + dS, H, t1, T2, r, b, v)
            - 2 * PartialTimeBarrier(TypeFlag, S, X, H, t1, T2, r, b, v)
            + PartialTimeBarrier(TypeFlag, S, X - dS, H, t1, T2, r, b, v)
        ) / (dS ^ 2)

    return EPartialTimeBarrier


# // Partial-time singel asset barrier options
def PartialTimeBarrier(
    TypeFlag: str,
    S: float,
    X: float,
    H: float,
    t1: float,
    T2: float,
    r: float,
    b: float,
    v: float,
):
    if TypeFlag == "cdoA":
        eta = 1
    elif TypeFlag == "cuoA":
        eta = -1

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T2) / (v * Sqr(T2))
    d2 = d1 - v * Sqr(T2)
    f1 = (Log(S / X) + 2 * Log(H / S) + (b + v ^ 2 / 2) * T2) / (v * Sqr(T2))
    f2 = f1 - v * Sqr(T2)
    e1 = (Log(S / H) + (b + v ^ 2 / 2) * t1) / (v * Sqr(t1))
    e2 = e1 - v * Sqr(t1)
    e3 = e1 + 2 * Log(H / S) / (v * Sqr(t1))
    e4 = e3 - v * Sqr(t1)
    mu = (b - v ^ 2 / 2) / v ^ 2
    rho = Sqr(t1 / T2)
    g1 = (Log(S / H) + (b + v ^ 2 / 2) * T2) / (v * Sqr(T2))
    g2 = g1 - v * Sqr(T2)
    g3 = g1 + 2 * Log(H / S) / (v * Sqr(T2))
    g4 = g3 - v * Sqr(T2)
    z1 = CND(e2) - (H / S) ^ (2 * mu) * CND(e4)
    z2 = CND(-e2) - (H / S) ^ (2 * mu) * CND(-e4)
    z3 = CBND(g2, e2, rho) - (H / S) ^ (2 * mu) * CBND(g4, -e4, -rho)
    z4 = CBND(-g2, -e2, rho) - (H / S) ^ (2 * mu) * CBND(-g4, e4, -rho)
    z5 = CND(e1) - (H / S) ^ (2 * (mu + 1)) * CND(e3)
    z6 = CND(-e1) - (H / S) ^ (2 * (mu + 1)) * CND(-e3)
    z7 = CBND(g1, e1, rho) - (H / S) ^ (2 * (mu + 1)) * CBND(g3, -e3, -rho)
    z8 = CBND(-g1, -e1, rho) - (H / S) ^ (2 * (mu + 1)) * CBND(-g3, e3, -rho)

    if (
        TypeFlag == "cdoA" or TypeFlag == "cuoA"
    ):  # // call down-and out and up-and-out type A
        PartialTimeBarrier = S * Exp((b - r) * T2) * (
            CBND(d1, eta * e1, eta * rho) - (H / S)
            ^ (2 * (mu + 1)) * CBND(f1, eta * e3, eta * rho)
        ) - X * Exp(-r * T2) * (
            CBND(d2, eta * e2, eta * rho) - (H / S)
            ^ (2 * mu) * CBND(f2, eta * e4, eta * rho)
        )
    elif TypeFlag == "cdoB2" and X < H:  # // call down-and-out type B2
        PartialTimeBarrier = S * Exp((b - r) * T2) * (
            CBND(g1, e1, rho) - (H / S) ^ (2 * (mu + 1)) * CBND(g3, -e3, -rho)
        ) - X * Exp(-r * T2) * (
            CBND(g2, e2, rho) - (H / S) ^ (2 * mu) * CBND(g4, -e4, -rho)
        )
    elif TypeFlag == "cdoB2" and X > H:
        PartialTimeBarrier = PartialTimeBarrier("coB1", S, X, H, t1, T2, r, b, v)
    elif TypeFlag == "cuoB2" and X < H:  # // call up-and-out type B2
        PartialTimeBarrier = (
            S
            * Exp((b - r) * T2)
            * (CBND(-g1, -e1, rho) - (H / S) ^ (2 * (mu + 1)) * CBND(-g3, e3, -rho))
            - X
            * Exp(-r * T2)
            * (CBND(-g2, -e2, rho) - (H / S) ^ (2 * mu) * CBND(-g4, e4, -rho))
            - S
            * Exp((b - r) * T2)
            * (CBND(-d1, -e1, rho) - (H / S) ^ (2 * (mu + 1)) * CBND(e3, -f1, -rho))
            + X
            * Exp(-r * T2)
            * (CBND(-d2, -e2, rho) - (H / S) ^ (2 * mu) * CBND(e4, -f2, -rho))
        )
    elif TypeFlag == "coB1" and X > H:  # // call out type B1
        PartialTimeBarrier = S * Exp((b - r) * T2) * (
            CBND(d1, e1, rho) - (H / S) ^ (2 * (mu + 1)) * CBND(f1, -e3, -rho)
        ) - X * Exp(-r * T2) * (
            CBND(d2, e2, rho) - (H / S) ^ (2 * mu) * CBND(f2, -e4, -rho)
        )
    elif TypeFlag == "coB1" and X < H:
        PartialTimeBarrier = (
            S
            * Exp((b - r) * T2)
            * (CBND(-g1, -e1, rho) - (H / S) ^ (2 * (mu + 1)) * CBND(-g3, e3, -rho))
            - X
            * Exp(-r * T2)
            * (CBND(-g2, -e2, rho) - (H / S) ^ (2 * mu) * CBND(-g4, e4, -rho))
            - S
            * Exp((b - r) * T2)
            * (CBND(-d1, -e1, rho) - (H / S) ^ (2 * (mu + 1)) * CBND(-f1, e3, -rho))
            + X
            * Exp(-r * T2)
            * (CBND(-d2, -e2, rho) - (H / S) ^ (2 * mu) * CBND(-f2, e4, -rho))
            + S
            * Exp((b - r) * T2)
            * (CBND(g1, e1, rho) - (H / S) ^ (2 * (mu + 1)) * CBND(g3, -e3, -rho))
            - X
            * Exp(-r * T2)
            * (CBND(g2, e2, rho) - (H / S) ^ (2 * mu) * CBND(g4, -e4, -rho))
        )
    elif TypeFlag == "pdoA":  # // put down-and out and up-and-out type A
        PartialTimeBarrier = (
            PartialTimeBarrier("cdoA", S, X, H, t1, T2, r, b, v)
            - S * Exp((b - r) * T2) * z5
            + X * Exp(-r * T2) * z1
        )
    elif TypeFlag == "puoA":
        PartialTimeBarrier = (
            PartialTimeBarrier("cuoA", S, X, H, t1, T2, r, b, v)
            - S * Exp((b - r) * T2) * z6
            + X * Exp(-r * T2) * z2
        )
    elif TypeFlag == "poB1":  # // put out type B1
        PartialTimeBarrier = (
            PartialTimeBarrier("coB1", S, X, H, t1, T2, r, b, v)
            - S * Exp((b - r) * T2) * z8
            + X * Exp(-r * T2) * z4
            - S * Exp((b - r) * T2) * z7
            + X * Exp(-r * T2) * z3
        )
    elif TypeFlag == "pdoB2":  # // put down-and-out type B2
        PartialTimeBarrier = (
            PartialTimeBarrier("cdoB2", S, X, H, t1, T2, r, b, v)
            - S * Exp((b - r) * T2) * z7
            + X * Exp(-r * T2) * z3
        )
    elif TypeFlag == "puoB2":  # // put up-and-out type B2
        PartialTimeBarrier = (
            PartialTimeBarrier("cuoB2", S, X, H, t1, T2, r, b, v)
            - S * Exp((b - r) * T2) * z8
            + X * Exp(-r * T2) * z4
        )
    return PartialTimeBarrier


# AquíEmpiezaAma
# // Double barrier options
def DoubleBarrier(
    TypeFlag: str,
    S: float,
    X: float,
    L: float,
    U: float,
    T: float,
    r: float,
    b: float,
    v: float,
    delta1: float,
    delta2: float,
) -> float:

    F = U * Exp(delta1 * T)
    E = L * Exp(delta2 * T)
    Sum1 = 0
    Sum2 = 0

    if TypeFlag in ["co", "ci"]:
        for n in range(-5, 5):
            d1 = (Log(S * U ^ (2 * n) / (X * L ^ (2 * n))) + (b + v ^ 2 / 2) * T) / (
                v * Sqr(T)
            )
            d2 = (Log(S * U ^ (2 * n) / (F * L ^ (2 * n))) + (b + v ^ 2 / 2) * T) / (
                v * Sqr(T)
            )
            d3 = (
                Log(L ^ (2 * n + 2) / (X * S * U ^ (2 * n))) + (b + v ^ 2 / 2) * T
            ) / (v * Sqr(T))
            d4 = (
                Log(L ^ (2 * n + 2) / (F * S * U ^ (2 * n))) + (b + v ^ 2 / 2) * T
            ) / (v * Sqr(T))
            mu1 = 2 * (b - delta2 - n * (delta1 - delta2)) / v ^ 2 + 1
            mu2 = 2 * n * (delta1 - delta2) / v ^ 2
            mu3 = 2 * (b - delta2 + n * (delta1 - delta2)) / v ^ 2 + 1
            Sum1 = (
                Sum1 + (U ^ n / L ^ n)
                ^ mu1 * (L / S)
                ^ mu2 * (CND(d1) - CND(d2)) - (L ^ (n + 1) / (U ^ n * S))
                ^ mu3 * (CND(d3) - CND(d4))
            )
            Sum2 = (
                Sum2 + (U ^ n / L ^ n)
                ^ (mu1 - 2) * (L / S)
                ^ mu2 * (CND(d1 - v * Sqr(T)) - CND(d2 - v * Sqr(T)))
                - (L ^ (n + 1) / (U ^ n * S))
                ^ (mu3 - 2) * (CND(d3 - v * Sqr(T)) - CND(d4 - v * Sqr(T)))
            )
        OutValue = S * Exp((b - r) * T) * Sum1 - X * Exp(-r * T) * Sum2

    elif TypeFlag in ["po", "pi"]:
        for n in range(-5, 5):
            d1 = (Log(S * U ^ (2 * n) / (E * L ^ (2 * n))) + (b + v ^ 2 / 2) * T) / (
                v * Sqr(T)
            )
            d2 = (Log(S * U ^ (2 * n) / (X * L ^ (2 * n))) + (b + v ^ 2 / 2) * T) / (
                v * Sqr(T)
            )
            d3 = (
                Log(L ^ (2 * n + 2) / (E * S * U ^ (2 * n))) + (b + v ^ 2 / 2) * T
            ) / (v * Sqr(T))
            d4 = (
                Log(L ^ (2 * n + 2) / (X * S * U ^ (2 * n))) + (b + v ^ 2 / 2) * T
            ) / (v * Sqr(T))
            mu1 = 2 * (b - delta2 - n * (delta1 - delta2)) / v ^ 2 + 1
            mu2 = 2 * n * (delta1 - delta2) / v ^ 2
            mu3 = 2 * (b - delta2 + n * (delta1 - delta2)) / v ^ 2 + 1
            Sum1 = (
                Sum1 + (U ^ n / L ^ n)
                ^ mu1 * (L / S)
                ^ mu2 * (CND(d1) - CND(d2)) - (L ^ (n + 1) / (U ^ n * S))
                ^ mu3 * (CND(d3) - CND(d4))
            )
            Sum2 = (
                Sum2 + (U ^ n / L ^ n)
                ^ (mu1 - 2) * (L / S)
                ^ mu2 * (CND(d1 - v * Sqr(T)) - CND(d2 - v * Sqr(T)))
                - (L ^ (n + 1) / (U ^ n * S))
                ^ (mu3 - 2) * (CND(d3 - v * Sqr(T)) - CND(d4 - v * Sqr(T)))
            )
        OutValue = X * Exp(-r * T) * Sum2 - S * Exp((b - r) * T) * Sum1

    if TypeFlag in ["co", "po"]:
        DoubleBarrier = OutValue
    elif TypeFlag == "ci":
        DoubleBarrier = GBlackScholes("c", S, X, T, r, b, v) - OutValue
    elif TypeFlag == "pi":
        DoubleBarrier = GBlackScholes("p", S, X, T, r, b, v) - OutValue


# Standard barrier options
def StandardBarrier(
    TypeFlag: str,
    S: float,
    X: float,
    H: float,
    k: float,
    T: float,
    r: float,
    b: float,
    v: float,
):

    # TypeFlag:      The "TypeFlag" gives you 8 different standard barrier
    #              1) "cdi"=Down-and-in call,    2) "cui"=Up-and-in call
    #               3) "pdi"=Down-and-in put,     4) "pui"=Up-and-in put
    #               5) "cdo"=Down-and-out call,   6) "cuo"=Up-out-in call
    #              7) "pdo"=Down-and-out put,    8) "puo"=Up-out-in put

    mu = (b - v ^ 2 / 2) / v ^ 2
    Lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
    X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
    X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
    y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
    y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
    z = Log(H / S) / (v * Sqr(T)) + Lambda * v * Sqr(T)

    if TypeFlag in ["cdi", "cdo"]:
        eta = 1
        phi = 1
    elif TypeFlag in ["cui", "cuo"]:
        eta = -1
        phi = 1
    elif TypeFlag in ["pdi", "pdo"]:
        eta = 1
        phi = -1
    elif TypeFlag in ["pui", "puo"]:
        eta = -1
        phi = -1

    f1 = phi * S * Exp((b - r) * T) * CND(phi * X1) - phi * X * Exp(-r * T) * CND(
        phi * X1 - phi * v * Sqr(T)
    )
    f2 = phi * S * Exp((b - r) * T) * CND(phi * X2) - phi * X * Exp(-r * T) * CND(
        phi * X2 - phi * v * Sqr(T)
    )
    f3 = (
        phi * S * Exp((b - r) * T) * (H / S)
        ^ (2 * (mu + 1)) * CND(eta * y1) - phi * X * Exp(-r * T) * (H / S)
        ^ (2 * mu) * CND(eta * y1 - eta * v * Sqr(T))
    )
    f4 = (
        phi * S * Exp((b - r) * T) * (H / S)
        ^ (2 * (mu + 1)) * CND(eta * y2) - phi * X * Exp(-r * T) * (H / S)
        ^ (2 * mu) * CND(eta * y2 - eta * v * Sqr(T))
    )
    f5 = (
        k
        * Exp(-r * T)
        * (
            CND(eta * X2 - eta * v * Sqr(T)) - (H / S)
            ^ (2 * mu) * CND(eta * y2 - eta * v * Sqr(T))
        )
    )
    f6 = k * (
        (H / S)
        ^ (mu + Lambda) * CND(eta * z) + (H / S)
        ^ (mu - Lambda) * CND(eta * z - 2 * eta * Lambda * v * Sqr(T))
    )

    if X > H:
        if TypeFlag == "cdi":  #'1a) cdi
            StandardBarrier = f3 + f5
        elif TypeFlag == "cui":  # 2a) cui
            StandardBarrier = f1 + f5
        elif TypeFlag == "pdi":  # 3a) pdi
            StandardBarrier = f2 - f3 + f4 + f5
        elif TypeFlag == "pui":  # 4a) pui
            StandardBarrier = f1 - f2 + f4 + f5
        elif TypeFlag == "cdo":  # 5a) cdo
            StandardBarrier = f1 - f3 + f6
        elif TypeFlag == "cuo":  # 6a) cuo
            StandardBarrier = f6
        elif TypeFlag == "pdo":  # 7a) pdo
            StandardBarrier = f1 - f2 + f3 - f4 + f6
        elif TypeFlag == "puo":  # 8a) puo
            StandardBarrier = f2 - f4 + f6

    elif X < H:
        if TypeFlag == "cdi":  # 1b) cdi
            StandardBarrier = f1 - f2 + f4 + f5
        elif TypeFlag == "cui":  # 2b) cui
            StandardBarrier = f2 - f3 + f4 + f5
        elif TypeFlag == "pdi":  # 3b) pdi
            StandardBarrier = f1 + f5
        elif TypeFlag == "pui":  # 4b) pui
            StandardBarrier = f3 + f5
        elif TypeFlag == "cdo":  # 5b) cdo
            StandardBarrier = f2 + f6 - f4
        elif TypeFlag == "cuo":  # 6b) cuo
            StandardBarrier = f1 - f2 + f3 - f4 + f6
        elif TypeFlag == "pdo":  # 7b) pdo
            StandardBarrier = f6
        elif TypeFlag == "puo":  # 8b) puo
            StandardBarrier = f1 - f3 + f6

    return StandardBarrier


def EDoubleBarrier(
    OutPutFlag: str,
    TypeFlag: str,
    S: float,
    X: float,
    L: float,
    U: float,
    T: float,
    r: float,
    b: float,
    v: float,
    delta1: float,
    delta2: float,
    dS=None,
):

    if dS is None:
        dS = 0.0001

    OutInnFlag = Right(TypeFlag, 1)
    CallPutFlag = Left(TypeFlag, 1)

    if OutInnFlag == "o" and (S <= L or S >= U):
        EDoubleBarrier = 0
        return EDoubleBarrier

    elif OutInnFlag == "i" and (S <= L or S >= U):
        EDoubleBarrier = EGBlackScholes(OutPutFlag, CallPutFlag, S, X, T, r, b, v)
        return EDoubleBarrier

    if OutPutFlag == "p":  # Value
        EDoubleBarrier = DoubleBarrier(TypeFlag, S, X, L, U, T, r, b, v, delta1, delta2)
    elif OutPutFlag == "d":  # Delta
        EDoubleBarrier = (
            DoubleBarrier(TypeFlag, S + dS, X, L, U, T, r, b, v, delta1, delta2)
            - DoubleBarrier(TypeFlag, S - dS, X, L, U, T, r, b, v, delta1, delta2)
        ) / (2 * dS)
    elif OutPutFlag == "dddv":  # DeltaDVol
        EDoubleBarrier = (
            1
            / (4 * dS * 0.01)
            * (
                DoubleBarrier(
                    TypeFlag, S + dS, X, L, U, T, r, b, v + 0.01, delta1, delta2
                )
                - DoubleBarrier(
                    TypeFlag, S + dS, X, L, U, T, r, b, v - 0.01, delta1, delta2
                )
                - DoubleBarrier(
                    TypeFlag, S - dS, X, L, U, T, r, b, v + 0.01, delta1, delta2
                )
                + DoubleBarrier(
                    TypeFlag, S - dS, X, L, U, T, r, b, v - 0.01, delta1, delta2
                )
            )
            / 100
        )
    elif OutPutFlag == "g":  # Gamma
        EDoubleBarrier = (
            DoubleBarrier(TypeFlag, S + dS, X, L, U, T, r, b, v, delta1, delta2)
            - 2 * DoubleBarrier(TypeFlag, S, X, L, U, T, r, b, v, delta1, delta2)
            + DoubleBarrier(TypeFlag, S - dS, X, L, U, T, r, b, v, delta1, delta2)
        ) / (dS ^ 2)
    elif OutPutFlag == "gp":  # GammaP
        EDoubleBarrier = (
            S
            / 100
            * EDoubleBarrier("g", TypeFlag, S + dS, X, L, U, T, r, b, v, delta1, delta2)
        )
    elif OutPutFlag == "gv":  # DGammaDVol
        EDoubleBarrier = (
            (
                DoubleBarrier(
                    TypeFlag, S + dS, X, L, U, T, r, b, v + 0.01, delta1, delta2
                )
                - 2
                * DoubleBarrier(TypeFlag, S, X, L, U, T, r, b, v + 0.01, delta1, delta2)
                + DoubleBarrier(
                    TypeFlag, S - dS, X, L, U, T, r, b, v + 0.01, delta1, delta2
                )
                - DoubleBarrier(
                    TypeFlag, S + dS, X, L, U, T, r, b, v - 0.01, delta1, delta2
                )
                + 2
                * DoubleBarrier(TypeFlag, S, X, L, U, T, r, b, v - 0.01, delta1, delta2)
                - DoubleBarrier(
                    TypeFlag, S - dS, X, L, U, T, r, b, v - 0.01, delta1, delta2
                )
            )
            / (2 * 0.01 * dS ^ 2)
            / 100
        )
    elif OutPutFlag == "v":  # Vega
        EDoubleBarrier = (
            DoubleBarrier(TypeFlag, S, X, L, U, T, r, b, v + 0.01, delta1, delta2)
            - DoubleBarrier(TypeFlag, S, X, L, U, T, r, b, v - 0.01, delta1, delta2)
        ) / 2
    elif OutPutFlag == "dvdv":  # DVegaDVol/Vomma
        EDoubleBarrier = (
            DoubleBarrier(TypeFlag, S, X, L, U, T, r, b, v + 0.01, delta1, delta2)
            - 2 * DoubleBarrier(TypeFlag, S, X, L, U, T, r, b, v, delta1, delta2)
            + DoubleBarrier(TypeFlag, S, X, L, U, T, r, b, v - 0.01, delta1, delta2)
        ) / 0.01 ^ 2 / 10000
    elif OutPutFlag == "vp":  # VegaP
        EDoubleBarrier = (
            v
            / 0.1
            * EDoubleBarrier("v", TypeFlag, S + dS, X, L, U, T, r, b, v, delta1, delta2)
        )
    elif OutPutFlag == "r":  # Rho
        EDoubleBarrier = (
            DoubleBarrier(
                TypeFlag, S, X, L, U, T, r + 0.01, b + 0.01, v, delta1, delta2
            )
            - DoubleBarrier(
                TypeFlag, S, X, L, U, T, r - 0.01, b - 0.01, v, delta1, delta2
            )
        ) / 2
    elif OutPutFlag == "fr":  # Futures option Rho
        EDoubleBarrier = (
            DoubleBarrier(TypeFlag, S, X, L, U, T, r + 0.01, 0, v, delta1, delta2)
            - DoubleBarrier(TypeFlag, S, X, L, U, T, r - 0.01, 0, v, delta1, delta2)
        ) / 2
    elif OutPutFlag == "f":  # Rho2/Phi
        EDoubleBarrier = (
            DoubleBarrier(TypeFlag, S, X, L, U, T, r, b - 0.01, v, delta1, delta2)
            - DoubleBarrier(TypeFlag, S, X, L, U, T, r, b + 0.01, v, delta1, delta2)
        ) / 2
    elif OutPutFlag == "b":  # Carry sensitivity
        EDoubleBarrier = (
            DoubleBarrier(TypeFlag, S, X, L, U, T, r, b + 0.01, v, delta1, delta2)
            - DoubleBarrier(TypeFlag, S, X, L, U, T, r, b - 0.01, v, delta1, delta2)
        ) / 2
    elif OutPutFlag == "t":  # Theta
        if T <= 1 / 365:
            EDoubleBarrier = DoubleBarrier(
                TypeFlag, S, X, L, U, 1e-05, r, b, v, delta1, delta2
            ) - DoubleBarrier(TypeFlag, S, X, L, U, T, r, b, v, delta1, delta2)

        else:
            EDoubleBarrier = DoubleBarrier(
                TypeFlag, S, X, L, U, T - 1 / 365, r, b, v, delta1, delta2
            ) - DoubleBarrier(TypeFlag, S, X, L, U, T, r, b, v, delta1, delta2)

    elif OutPutFlag == "dx":  # Strike Delta
        EDoubleBarrier = (
            DoubleBarrier(TypeFlag, S, X + dS, L, U, T, r, b, v, delta1, delta2)
            - DoubleBarrier(TypeFlag, S, X - dS, L, U, T, r, b, v, delta1, delta2)
        ) / (2 * dS)
    elif OutPutFlag == "dxdx":  # Strike Gamma
        EDoubleBarrier = (
            DoubleBarrier(TypeFlag, S, X + dS, L, U, T, r, b, v, delta1, delta2)
            - 2 * DoubleBarrier(TypeFlag, S, X, L, U, T, r, b, v, delta1, delta2)
            + DoubleBarrier(TypeFlag, S, X - dS, L, U, T, r, b, v, delta1, delta2)
        ) / (dS ^ 2)
    return EDoubleBarrier


def IsMissing(dS):
    return dS is None


def EStandardBarrier(
    OutPutFlag: str,
    TypeFlag: str,
    S: float,
    X: float,
    H: float,
    k: float,
    T: float,
    r: float,
    b: float,
    v: float,
    dS=None,
):

    if IsMissing(dS):
        dS = 0.0001

    OutInnFlag = Right(TypeFlag, 2)
    CallPutFlag = Left(TypeFlag, 1)

    if (OutInnFlag == "do" and S <= H) or (OutInnFlag == "uo" and S >= H):
        if OutPutFlag == "p":
            EStandardBarrier = k
        else:
            EStandardBarrier = 0
        return EStandardBarrier

    elif (OutInnFlag == "di" and S <= H) or (OutInnFlag == "ui" and S >= H):
        EStandardBarrier = EGBlackScholes(OutPutFlag, CallPutFlag, S, X, T, r, b, v)
        return EStandardBarrier

    if OutPutFlag == "p":  # Value
        EStandardBarrier == StandardBarrier(TypeFlag, S, X, H, k, T, r, b, v)
    elif OutPutFlag == "d":  # Delta
        EStandardBarrier = (
            StandardBarrier(TypeFlag, S + dS, X, H, k, T, r, b, v)
            - StandardBarrier(TypeFlag, S - dS, X, H, k, T, r, b, v)
        ) / (2 * dS)
    elif OutPutFlag == "dddv":  # DeltaDVol
        EStandardBarrier = (
            1
            / (4 * dS * 0.01)
            * (
                StandardBarrier(TypeFlag, S + dS, X, H, k, T, r, b, v + 0.01)
                - StandardBarrier(TypeFlag, S + dS, X, H, k, T, r, b, v - 0.01)
                - StandardBarrier(TypeFlag, S - dS, X, H, k, T, r, b, v + 0.01)
                + StandardBarrier(TypeFlag, S - dS, X, H, k, T, r, b, v - 0.01)
            )
            / 100
        )
    elif OutPutFlag == "g":  # Gamma
        EStandardBarrier = (
            StandardBarrier(TypeFlag, S + dS, X, H, k, T, r, b, v)
            - 2 * StandardBarrier(TypeFlag, S, X, H, k, T, r, b, v)
            + StandardBarrier(TypeFlag, S - dS, X, H, k, T, r, b, v)
        ) / (dS ^ 2)
    elif OutPutFlag == "gp":  # GammaP
        EStandardBarrier = (
            S / 100 * EStandardBarrier("g", TypeFlag, S + dS, X, H, k, T, r, b, v)
        )
    elif OutPutFlag == "gv":  # DGammaDvol
        EStandardBarrier = (
            (
                StandardBarrier(TypeFlag, S + dS, X, H, k, T, r, b, v + 0.01)
                - 2 * StandardBarrier(TypeFlag, S, X, H, k, T, r, b, v + 0.01)
                + StandardBarrier(TypeFlag, S - dS, X, H, k, T, r, b, v + 0.01)
                - StandardBarrier(TypeFlag, S + dS, X, H, k, T, r, b, v - 0.01)
                + 2 * StandardBarrier(TypeFlag, S, X, H, k, T, r, b, v - 0.01)
                - StandardBarrier(TypeFlag, S - dS, X, H, k, T, r, b, v - 0.01)
            )
            / (2 * 0.01 * dS ^ 2)
            / 100
        )
    elif OutPutFlag == "v":  # Vega
        EStandardBarrier = (
            StandardBarrier(TypeFlag, S, X, H, k, T, r, b, v + 0.01)
            - StandardBarrier(TypeFlag, S, X, H, k, T, r, b, v - 0.01)
        ) / 2
    elif OutPutFlag == "vp":  # VegaP
        EStandardBarrier = (
            v / 0.1 * EStandardBarrier("v", TypeFlag, S + dS, X, H, k, T, r, b, v)
        )
    elif OutPutFlag == "dvdv":  # DvegaDvol/vomma
        EStandardBarrier = (
            StandardBarrier(TypeFlag, S, X, H, k, T, r, b, v + 0.01)
            - 2 * StandardBarrier(TypeFlag, S, X, H, k, T, r, b, v)
            + StandardBarrier(TypeFlag, S, X, H, k, T, r, b, v - 0.01)
        ) / 0.01 ^ 2 / 10000
    elif OutPutFlag == "r":  # Rho
        EStandardBarrier = (
            StandardBarrier(TypeFlag, S, X, H, k, T, r + 0.01, b + 0.01, v)
            - StandardBarrier(TypeFlag, S, X, H, k, T, r - 0.01, b - 0.01, v)
        ) / 2
    elif OutPutFlag == "fr":  # Futures option Rho
        EStandardBarrier = (
            StandardBarrier(TypeFlag, S, X, H, k, T, r + 0.01, 0, v)
            - StandardBarrier(TypeFlag, S, X, H, k, T, r - 0.01, 0, v)
        ) / 2
    elif OutPutFlag == "f":  # Rho2 Phi
        EStandardBarrier = (
            StandardBarrier(TypeFlag, S, X, H, k, T, r, b - 0.01, v)
            - StandardBarrier(TypeFlag, S, X, H, k, T, r, b + 0.01, v)
        ) / 2
    elif OutPutFlag == "b":  # Carry sensitivity
        EStandardBarrier = (
            StandardBarrier(TypeFlag, S, X, H, k, T, r, b + 0.01, v)
            - StandardBarrier(TypeFlag, S, X, H, k, T, r, b - 0.01, v)
        ) / 2
    elif OutPutFlag == "t":  # Theta
        if T <= 1 / 365:
            EStandardBarrier = StandardBarrier(
                TypeFlag, S, X, H, k, 1e-05, r, b, v
            ) - StandardBarrier(TypeFlag, S, X, H, k, T, r, b, v)
        else:
            EStandardBarrier = StandardBarrier(
                TypeFlag, S, X, H, k, T - 1 / 365, r, b, v
            ) - StandardBarrier(TypeFlag, S, X, H, k, T, r, b, v)
    elif OutPutFlag == "dx":  # Strike Delta
        EStandardBarrier = (
            StandardBarrier(TypeFlag, S, X + dS, H, k, T, r, b, v)
            - StandardBarrier(TypeFlag, S, X - dS, H, k, T, r, b, v)
        ) / (2 * dS)
    elif OutPutFlag == "dxdx":  # Strike Gamma
        EStandardBarrier = (
            StandardBarrier(TypeFlag, S, X + dS, H, k, T, r, b, v)
            - 2 * StandardBarrier(TypeFlag, S, X, H, k, T, r, b, v)
            + StandardBarrier(TypeFlag, S, X - dS, H, k, T, r, b, v)
        ) / (dS ^ 2)

    return EStandardBarrier


# // Discrete barrier monitoring adjustment
def DiscreteAdjustedBarrier(S: float, H: float, v: float, dt: float):
    if H > S:
        DiscreteAdjustedBarrier = H * Exp(0.5826 * v * Sqr(dt))
    else:
        DiscreteAdjustedBarrier = H * Exp(-0.5826 * v * Sqr(dt))
    return DiscreteAdjustedBarrier


def CND(X: float):
    y = abs(X)
    if y > 37:
        CND = 0
    else:
        Exponential = Exp(-y ^ 2 / 2)
        if y < 7.07106781186547:
            SumA = 0.0352624965998911 * y + 0.700383064443688
            SumA = SumA * y + 6.37396220353165
            SumA = SumA * y + 33.912866078383
            SumA = SumA * y + 112.079291497871
            SumA = SumA * y + 221.213596169931
            SumA = SumA * y + 220.206867912376
            SumB = 0.0883883476483184 * y + 1.75566716318264
            SumB = SumB * y + 16.064177579207
            SumB = SumB * y + 86.7807322029461
            SumB = SumB * y + 296.564248779674
            SumB = SumB * y + 637.333633378831
            SumB = SumB * y + 793.826512519948
            SumB = SumB * y + 440.413735824752
            CND = Exponential * SumA / SumB
        else:
            SumA = y + 0.65
            SumA = y + 4 / SumA
            SumA = y + 3 / SumA
            SumA = y + 2 / SumA
            SumA = y + 1 / SumA
            CND = Exponential / (SumA * 2.506628274631)
    if X > 0:
        CND = 1 - CND
    return CND


def Sin(value: float):
    return np.sin(value)


def XX(i, j):
    xx = {
        1: {1: -0.932469514203152, 2: -0.981560634246719, 3: -0.993128599185095},
        2: {1: -0.661209386466265, 2: -0.904117256370475, 3: -0.963971927277914},
        3: {1: -0.238619186083197, 2: -0.769902674194305, 3: -0.912234428251326},
        4: {2: -0.587317954286617, 3: -0.839116971822219},
        5: {2: -0.36783149899818, 3: -0.746331906460151},
        6: {2: -0.125233408511469, 3: -0.636053680726515},
        7: {3: -0.510867001950827},
        8: {3: -0.37370608871542},
        9: {3: -0.227785851141645},
        10: {3: -0.0765265211334973},
    }
    return xx[i][j]


def W(i, j):
    w = {
        1: {1: 0.17132449237917, 2: 0.0471753363865118, 3: 0.0176140071391521},
        2: {1: 0.360761573048138, 2: 0.106939325995318, 3: 0.0406014298003869},
        3: {1: 0.46791393457269, 2: 0.160078328543346, 3: 0.0626720483341091},
        4: {2: 0.203167426723066, 3: 0.0832767415767048},
        5: {2: 0.233492536538355, 3: 0.10193011981724},
        6: {2: 0.249147045813403, 3: 0.118194531961518},
        7: {3: 0.131688638449177},
        8: {3: 0.142096109318382},
        9: {3: 0.149172986472604},
        10: {3: 0.152753387130726},
    }
    return w[i][j]


# // The cumulative bivariate normal distribution function
def CBND(X: float, y: float, rho: float):
    if abs(rho) < 0.3:
        NG = 1
        LG = 3
    elif abs(rho) < 0.75:
        NG = 2
        LG = 6
    else:
        LG = 10

    H = -X
    k = -y
    hk = H * k
    BVN = 0

    if abs(rho) < 0.925:
        if abs(rho) > 0:
            hs = (H * H + k * k) / 2
            asr = ArcSin(rho)
        for i in range(1, LG):
            for ISs in range(-1, 1, 2):
                sn = Sin(asr * (ISs * XX(i, NG) + 1) / 2)
                BVN = BVN + W(i, NG) * Exp((sn * hk - hs) / (1 - sn * sn))

        BVN = BVN * asr / (4 * np.pi)
        BVN = BVN + CND(-H) * CND(-k)
    else:
        if rho < 0:
            k = -k
            hk = -hk

    if abs(rho) < 1:
        Ass = (1 - rho) * (1 + rho)
        A = Sqr(Ass)
        bs = (H - k) ^ 2
        c = (4 - hk) / 8
        d = (12 - hk) / 16
        asr = -(bs / Ass + hk) / 2
        if asr > -100:
            BVN = (
                A
                * Exp(asr)
                * (1 - c * (bs - Ass) * (1 - d * bs / 5) / 3 + c * d * Ass * Ass / 5)
            )
        if -hk < 100:
            b = Sqr(bs)
            BVN = BVN - Exp(-hk / 2) * Sqr(2 * np.pi) * CND(-b / A) * b * (
                1 - c * bs * (1 - d * bs / 5) / 3
            )

        A = A / 2
        for i in range(1, LG):
            for ISs in range(-1, 1, 2):
                xs = (A * (ISs * XX(i, NG) + 1)) ^ 2
                rs = Sqr(1 - xs)
                asr = -(bs / xs + hk) / 2
                if asr > -100:
                    BVN = BVN + A * W(i, NG) * Exp(asr) * (
                        Exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs
                        - (1 + c * xs * (1 + d * xs))
                    )

        BVN = -BVN / (2 * np.pi)

    if rho > 0:
        BVN = BVN + CND(-Max(H, k))
    else:
        BVN = -BVN
        if k > H:
            BVN = BVN + CND(k) - CND(H)

    CBND = BVN
    return CBND


# %%
# // This is the generlaized Black-Scholes-Merton formula including all greeeks
# // This function is simply calling all the other functions
def EGBlackScholes(
    OutPutFlag: str,
    CallPutFlag: str = None,
    S: float = None,
    X: float = None,
    T: float = None,
    r: float = None,
    b: float = None,
    v: float = None,
    delta: float = None,
    InTheMoneyProb: float = None,
    ThetaDays: float = None,
) -> float:
    output = 0

    if OutPutFlag == "p":  # Value
        EGBlackScholes = GBlackScholes(CallPutFlag, S, X, T, r, b, v)

    # DELTA GREEKS
    elif OutPutFlag == "d":  # Delta
        EGBlackScholes = GDelta(CallPutFlag, S, X, T, r, b, v)
    elif OutPutFlag == "df":  # Forward Delta
        EGBlackScholes = GForwardDelta(CallPutFlag, S, X, T, r, b, v)
    elif OutPutFlag == "dddv":  # DDeltaDvol
        EGBlackScholes = GDdeltaDvol(S, X, T, r, b, v) / 100
    elif OutPutFlag == "dvv":  # DDeltaDvolDvol
        EGBlackScholes = GDdeltaDvolDvol(S, X, T, r, b, v) / 10000
    elif OutPutFlag == "dt":  # DDeltaDtime/Charm
        EGBlackScholes = GDdeltaDtime(CallPutFlag, S, X, T, r, b, v) / 365
    elif OutPutFlag == "dmx":
        EGBlackScholes = S ^ 2 / X * Exp((2 * b + v ^ 2) * T)
    elif OutPutFlag == "e":  # Elasticity
        EGBlackScholes = GElasticity(CallPutFlag, S, X, T, r, b, v)

    # GAMMA GREEKS
    elif OutPutFlag == "sg":  # SaddleGamma
        EGBlackScholes = GSaddleGamma(X, T, r, b, v)
    elif OutPutFlag == "g":  # Gamma
        EGBlackScholes = GGamma(S, X, T, r, b, v)
    elif OutPutFlag == "s":  # DgammaDspot/speed
        EGBlackScholes = GDgammaDspot(S, X, T, r, b, v)
    elif OutPutFlag == "gv":  # DgammaDvol/Zomma
        EGBlackScholes = GDgammaDvol(S, X, T, r, b, v) / 100
    elif OutPutFlag == "gt":  # DgammaDtime
        EGBlackScholes = GDgammaDtime(S, X, T, r, b, v) / 365

    elif OutPutFlag == "gp":  # GammaP
        EGBlackScholes = GGammaP(S, X, T, r, b, v)
    elif OutPutFlag == "gps":  # DgammaPDspot
        EGBlackScholes = GDgammaPDspot(S, X, T, r, b, v)
    elif OutPutFlag == "gpv":  # DgammaDvol/Zomma
        EGBlackScholes = GDgammaPDvol(S, X, T, r, b, v) / 100
    elif OutPutFlag == "gpt":  # DgammaPDtime
        EGBlackScholes = GDgammaPDtime(S, X, T, r, b, v) / 365

    # VEGA GREEKS
    elif OutPutFlag == "v":  # Vega
        EGBlackScholes = GVega(S, X, T, r, b, v) / 100
    elif OutPutFlag == "vt":  # DvegaDtime
        EGBlackScholes = GDvegaDtime(S, X, T, r, b, v) / 365
    elif OutPutFlag == "dvdv":  # DvegaDvol/Vomma
        EGBlackScholes = GDvegaDvol(S, X, T, r, b, v) / 10000
    elif OutPutFlag == "vvv":  # DvommaDvol
        EGBlackScholes = GDvommaDvol(S, X, T, r, b, v) / 1000000

    elif OutPutFlag == "vp":  # VegaP
        EGBlackScholes = GVegaP(S, X, T, r, b, v)
    elif OutPutFlag == "vpv":  # DvegaPDvol/VommaP
        EGBlackScholes = GDvegaPDvol(S, X, T, r, b, v) / 100
    elif OutPutFlag == "vl":  # Vega Leverage
        EGBlackScholes = GVegaLeverage(CallPutFlag, S, X, T, r, b, v)

    # VARIANCE GREEKS
    elif OutPutFlag == "varvega":  # Variance-Vega
        EGBlackScholes = GVarianceVega(S, X, T, r, b, v) / 100
    elif OutPutFlag == "vardelta":  # Variance-delta
        EGBlackScholes = GVarianceDelta(S, X, T, r, b, v) / 100
    elif OutPutFlag == "varvar":  # Variance-vomma
        EGBlackScholes = GVarianceVomma(S, X, T, r, b, v) / 10000

    # THETA GREEKS
    elif OutPutFlag == "t":  # Theta
        EGBlackScholes = GTheta(CallPutFlag, S, X, T, r, b, v) / 365
    elif OutPutFlag == "Dlt":  # Drift-less Theta
        EGBlackScholes = GThetaDriftLess(S, X, T, r, b, v) / 365

    # RATE/CARRY GREEKS
    elif OutPutFlag == "r":  # Rho
        EGBlackScholes = GRho(CallPutFlag, S, X, T, r, b, v) / 100
    elif OutPutFlag == "fr":  # Rho futures option
        EGBlackScholes = GRhoFO(CallPutFlag, S, X, T, r, b, v) / 100
    elif OutPutFlag == "b":  # Carry Rho
        EGBlackScholes = GCarry(CallPutFlag, S, X, T, r, b, v) / 100
    elif OutPutFlag == "f":  # Phi/Rho2
        EGBlackScholes = GPhi(CallPutFlag, S, X, T, r, b, v) / 100

    # PROB GREEKS
    elif OutPutFlag == "z":  # Zeta/In-the-money risk neutral probability
        EGBlackScholes = GInTheMoneyProbability(CallPutFlag, S, X, T, b, v)
    elif OutPutFlag == "zv":  # DzetaDvol
        EGBlackScholes = GDzetaDvol(CallPutFlag, S, X, T, r, b, v) / 100
    elif OutPutFlag == "zt":  # DzetaDtime
        EGBlackScholes = GDzetaDtime(CallPutFlag, S, X, T, r, b, v) / 365
    elif OutPutFlag == "bp":  # Brak even probability
        EGBlackScholes = GBreakEvenProbability(CallPutFlag, S, X, T, r, b, v)
    elif OutPutFlag == "dx":  # StrikeDelta
        EGBlackScholes = GStrikeDelta(CallPutFlag, S, X, T, r, b, v)
    elif OutPutFlag == "dxdx":  # Risk Neutral Density
        EGBlackScholes = GRiskNeutralDensity(S, X, T, r, b, v)

    # FROM DELTA GREEKS
    elif OutPutFlag == "gfd":  # Gamma from delta
        EGBlackScholes = GGammaFromDelta(S, T, r, b, v, delta)
    elif OutPutFlag == "gpfd":  # GammaP from delta
        EGBlackScholes = GGammaPFromDelta(S, T, r, b, v, delta)
    elif OutPutFlag == "vfd":  # Vega from delta
        EGBlackScholes = GVegaFromDelta(S, T, r, b, delta) / 100
    elif OutPutFlag == "vpfd":  # VegaP from delta
        EGBlackScholes = GVegaPFromDelta(S, T, r, b, v, delta)
    elif OutPutFlag == "xfd":  # Strike from delta
        EGBlackScholes = GStrikeFromDelta(CallPutFlag, S, T, r, b, v, delta)
    elif OutPutFlag == "ipfd":  # In-the-money probability from delta
        EGBlackScholes = InTheMoneyProbFromDelta(CallPutFlag, S, T, r, b, v, delta)

        # FROM IN-THE GREEKS
    elif OutPutFlag == "xfip":  # Strike from in-the-money probability
        EGBlackScholes = GStrikeFromInTheMoneyProb(
            CallPutFlag, S, v, T, b, InTheMoneyProb
        )
    elif OutPutFlag == "RNDfip":  # Risk Neutral Density from in-the-money probability
        EGBlackScholes = GRNDFromInTheMoneyProb(X, T, r, v, InTheMoneyProb)
    elif OutPutFlag == "dfip":  # Strike from in-the-money probability
        EGBlackScholes = GDeltaFromInTheMoneyProb(
            CallPutFlag, S, T, r, b, v, InTheMoneyProb
        )

    # CALCULATIONS
    elif OutPutFlag == "d1":  # d1
        EGBlackScholes = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    elif OutPutFlag == "d2":  # d2
        EGBlackScholes = (Log(S / X) + (b - v ^ 2 / 2) * T) / (v * Sqr(T))
    elif OutPutFlag == "nd1":  # n(d1)
        EGBlackScholes = ND((Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T)))
    elif OutPutFlag == "nd2":  # n(d2)
        EGBlackScholes = ND((Log(S / X) + (b - v ^ 2 / 2) * T) / (v * Sqr(T)))
    elif OutPutFlag == "CNDd1":  # N(d1)
        EGBlackScholes = CND((Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T)))
    elif OutPutFlag == "CNDd2":  # N(d2)
        EGBlackScholes = CND((Log(S / X) + (b - v ^ 2 / 2) * T) / (v * Sqr(T)))
    return EGBlackScholes


def BisectionAlgorithm(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, cm: float
):
    vLow = 0.01
    vHigh = 2
    epsilon = 1e-08
    cLow = GBlackScholes(CallPutFlag, S, X, T, r, b, vLow)
    cHigh = GBlackScholes(CallPutFlag, S, X, T, r, b, vHigh)

    counter = 0
    vi = vLow + (cm - cLow) * (vHigh - vLow) / (cHigh - cLow)
    while abs(cm - GBlackScholes(CallPutFlag, S, X, T, r, b, vi)) > epsilon:
        if counter > 100:
            BisectionAlgorithm = -1
            return BisectionAlgorithm

        if GBlackScholes(CallPutFlag, S, X, T, r, b, vi) < cm:
            vLow = vi
        else:
            vHigh = vi

        cLow = GBlackScholes(CallPutFlag, S, X, T, r, b, vLow)
        cHigh = GBlackScholes(CallPutFlag, S, X, T, r, b, vHigh)
        vi = vLow + (cm - cLow) * (vHigh - vLow) / (cHigh - cLow)

    BisectionAlgorithm = vi
    return BisectionAlgorithm


# Black and Scholes (1973) Stock options, on non dividend paying stock
def BlackScholes(CallPutFlag: str, S: float, X: float, T: float, r: float, v: float):

    d1 = (Log(S / X) + (r + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    if CallPutFlag == "c":
        BlackScholes = S * CND(d1) - X * Exp(-r * T) * CND(d2)
    elif CallPutFlag == "p":
        BlackScholes = X * Exp(-r * T) * CND(-d2) - S * CND(-d1)
    return BlackScholes


# Merton (1973) Options on stock indices paying continuous dividend yield q
def Merton73(
    CallPutFlag: str, S: float, X: float, T: float, r: float, q: float, v: float
):
    d1 = (Log(S / X) + (r - q + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    if CallPutFlag == "c":
        Merton73 = S * Exp(-q * T) * CND(d1) - X * Exp(-r * T) * CND(d2)
    elif CallPutFlag == "p":
        Merton73 = X * Exp(-r * T) * CND(-d2) - S * Exp(-q * T) * CND(-d1)
    return Merton73


# Black (1976) Options on futures/forwards
def Black76(CallPutFlag: str, f: float, X: float, T: float, r: float, v: float):
    d1 = (Log(f / X) + (v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    if CallPutFlag == "c":
        Black76 = Exp(-r * T) * (f * CND(d1) - X * CND(d2))
    elif CallPutFlag == "p":
        Black76 = Exp(-r * T) * (X * CND(-d2) - f * CND(-d1))
    return Black76


# Garman and Kohlhagen (1983) Currency options
def GarmanKolhagen(
    CallPutFlag: str, S: float, X: float, T: float, r: float, rf: float, v: float
):
    d1 = (Log(S / X) + (r - rf + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    if CallPutFlag == "c":
        GarmanKolhagen = S * Exp(-rf * T) * CND(d1) - X * Exp(-r * T) * CND(d2)
    elif CallPutFlag == "p":
        GarmanKolhagen = X * Exp(-r * T) * CND(-d2) - S * Exp(-rf * T) * CND(-d1)
    return GarmanKolhagen


#  The generalized Black and Scholes formula
def GBlackScholes(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
):
    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)

    if CallPutFlag == "c":
        GBlackScholes = S * Exp((b - r) * T) * CND(d1) - X * Exp(-r * T) * CND(d2)
    elif CallPutFlag == "p":
        GBlackScholes = X * Exp(-r * T) * CND(-d2) - S * Exp((b - r) * T) * CND(-d1)
    return GBlackScholes


# This is the generlaized Black-Scholes-Merton formula including all greeeks# This function is simply calling all the other functions


# DDeltaDvol also known as vanna
def GDdeltaDvol(S: float, X: float, T: float, r: float, b: float, v: float):
    d1 = (Log(S / X) + (b + v * v / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    GDdeltaDvol = -Exp((b - r) * T) * d2 / v * ND(d1)
    return GDdeltaDvol


# DDeltaDvolDvol also known as DVannaDvol
def GDdeltaDvolDvol(S: float, X: float, T: float, r: float, b: float, v: float):
    d1 = (Log(S / X) + (b + v * v / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    GDdeltaDvolDvol = GDdeltaDvol(S, X, T, r, b, v) * 1 / v * (d1 * d2 - d1 / d2 - 1)
    return GDdeltaDvolDvol


# Vega from delta
def GVegaFromDelta(S: float, T: float, r: float, b: float, delta: float):
    GVegaFromDelta = (
        S * Exp((b - r) * T) * Sqr(T) * ND(CNDEV(Exp((r - b) * T) * abs(delta)))
    )
    return GVegaFromDelta


# Gamma from delta
def GGammaFromDelta(S: float, T: float, r: float, b: float, v: float, delta: float):
    GGammaFromDelta = (
        Exp((b - r) * T) * ND(CNDEV(Exp((r - b) * T) * abs(delta))) / (S * v * Sqr(T))
    )
    return GGammaFromDelta


# Risk Neutral Density from in-the-money probability
def GRNDFromInTheMoneyProb(X: float, T: float, r: float, v: float, Probability: float):

    GRNDFromInTheMoneyProb = Exp(-r * T) * ND(CNDEV(Probability)) / (X * v * Sqr(T))
    return GRNDFromInTheMoneyProb


# GammaP from delta
def GGammaPFromDelta(
    S: float, T: float, r: float, b: float, v: float, delta: float
) -> float:
    GGammaPFromDelta = S / 100 * GGammaFromDelta(S, T, r, b, v, delta)
    return GGammaPFromDelta


# VegaP from delta
def GVegaPFromDelta(
    S: float, T: float, r: float, b: float, v: float, delta: float
) -> float:
    GVegaPFromDelta = v / 10 * GVegaFromDelta(S, T, r, b, delta)
    return GVegaPFromDelta


# What asset price that gives maximum DdeltaDvol
def MaxDdeltaDvolAsset(
    UpperLowerFlag: str, X: float, T: float, b: float, v: float
) -> float:
    # UpperLowerFlag"l" gives lower asset level that gives max DdeltaDvol
    # UpperLowerFlag"l" gives upper asset level that gives max DdeltaDvol

    if UpperLowerFlag == "l":
        MaxDdeltaDvolAsset = X * Exp(-b * T - v * Sqr(T) * Sqr(4 + T * v ^ 2) / 2)
    elif UpperLowerFlag == "u":
        MaxDdeltaDvolAsset = X * Exp(-b * T + v * Sqr(T) * Sqr(4 + T * v ^ 2) / 2)
    return MaxDdeltaDvolAsset


# What strike price that gives maximum DdeltaDvol
def MaxDdeltaDvolStrike(
    UpperLowerFlag: str, S: float, T: float, b: float, v: float
) -> float:

    # UpperLowerFlag"l" gives lower strike level that gives max DdeltaDvol
    # UpperLowerFlag"l" gives upper strike level that gives max DdeltaDvol

    if UpperLowerFlag == "l":
        MaxDdeltaDvolStrike = S * Exp(b * T - v * Sqr(T) * Sqr(4 + T * v ^ 2) / 2)
    elif UpperLowerFlag == "u":
        MaxDdeltaDvolStrike = S * Exp(b * T + v * Sqr(T) * Sqr(4 + T * v ^ 2) / 2)
    return MaxDdeltaDvolStrike


# What strike price that gives maximum gamma and vega
def GMaxGammaVegaatX(S: float, b: float, T: float, v: float):

    GMaxGammaVegaatX = S * Exp((b + v * v / 2) * T)

    return GMaxGammaVegaatX


# What asset price that gives maximum gamma
def GMaxGammaatS(X: float, b: float, T: float, v: float):

    GMaxGammaatS = X * Exp((-b - 3 * v * v / 2) * T)

    return GMaxGammaatS


# What asset price that gives maximum vega
def GMaxVegaatS(X: float, b: float, T: float, v: float):

    GMaxVegaatS = X * Exp((-b + v * v / 2) * T)

    return GMaxVegaatS


# Forward delta for the generalized Black and Scholes formula
def GForwardDelta(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
) -> float:

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))

    if CallPutFlag == "c":
        GForwardDelta = Exp(-r * T) * CND(d1)
    elif CallPutFlag == "p":
        GForwardDelta = Exp(-r * T) * (CND(d1) - 1)
    return GForwardDelta


# DZetaDvol for the generalized Black and Scholes formula
def GDzetaDvol(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
) -> float:
    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    if CallPutFlag == "c":
        GDzetaDvol = -ND(d2) * d1 / v
    else:
        GDzetaDvol = ND(d2) * d1 / v
    return GDzetaDvol


# DZetaDtime for the generalized Black and Scholes formula
def GDzetaDtime(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
) -> float:

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    if CallPutFlag == "c":
        GDzetaDtime = ND(d2) * (b / (v * Sqr(T)) - d1 / (2 * T))
    else:
        GDzetaDtime = -ND(d2) * (b / (v * Sqr(T)) - d1 / (2 * T))
    return GDzetaDtime


# Delta for the generalized Black and Scholes formula
def GInTheMoneyProbability(
    CallPutFlag: str, S: float, X: float, T: float, b: float, v: float
) -> float:
    d2 = (Log(S / X) + (b - v ^ 2 / 2) * T) / (v * Sqr(T))

    if CallPutFlag == "c":
        GInTheMoneyProbability = CND(d2)
    elif CallPutFlag == "p":
        GInTheMoneyProbability = CND(-d2)
    return GInTheMoneyProbability


# Risk neutral break even probability for the generalized Black and Scholes formula
def GBreakEvenProbability(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
) -> float:

    if CallPutFlag == "c":
        X = X + GBlackScholes("c", S, X, T, r, b, v) * Exp(r * T)
        d2 = (Log(S / X) + (b - v ^ 2 / 2) * T) / (v * Sqr(T))
        GBreakEvenProbability = CND(d2)
    elif CallPutFlag == "p":
        X = X - GBlackScholes("p", S, X, T, r, b, v) * Exp(r * T)
        d2 = (Log(S / X) + (b - v ^ 2 / 2) * T) / (v * Sqr(T))
        GBreakEvenProbability = CND(-d2)

    return GBreakEvenProbability


## Aquí empezó Ama
## Closed form solution to find strike given the in-the-money risk neutral probability
def GStrikeFromInTheMoneyProb(
    CallPutFlag: str, S: float, v: float, T: float, b: float, InTheMoneyProb: float
):

    if CallPutFlag == "c":
        GStrikeFromInTheMoneyProb = S * Exp(
            -CNDEV(InTheMoneyProb) * v * Sqr(T) + (b - v * v / 2) * T
        )
    else:
        GStrikeFromInTheMoneyProb = S * Exp(
            CNDEV(InTheMoneyProb) * v * Sqr(T) + (b - v * v / 2) * T
        )
    return GStrikeFromInTheMoneyProb


# Closed form solution to find strike given the delta
def GStrikeFromDelta(
    CallPutFlag: str, S: float, T: float, r: float, b: float, v: float, delta: float
):

    if CallPutFlag == "c":
        GStrikeFromDelta = S * Exp(
            -CNDEV(delta * Exp((r - b) * T)) * v * Sqr(T) + (b + v * v / 2) * T
        )
    else:
        GStrikeFromDelta = S * Exp(
            CNDEV(-delta * Exp((r - b) * T)) * v * Sqr(T) + (b + v * v / 2) * T
        )
    return GStrikeFromDelta


# Closed form solution to find in-the-money risk-neutral probaility given the delta
def InTheMoneyProbFromDelta(
    CallPutFlag: str, S: float, T: float, r: float, b: float, v: float, delta: float
):

    if CallPutFlag == "c":
        InTheMoneyProbFromDelta = CND(CNDEV(delta / Exp((b - r) * T)) - v * Sqr(T))
    else:
        InTheMoneyProbFromDelta = CND(CNDEV(-delta / Exp((b - r) * T)) + v * Sqr(T))
    return InTheMoneyProbFromDelta


# Closed form solution to find in-the-money risk-neutral probaility given the delta
def GDeltaFromInTheMoneyProb(
    CallPutFlag: str,
    S: float,
    T: float,
    r: float,
    b: float,
    v: float,
    InTheMoneyProb: float,
):

    if CallPutFlag == "c":
        GDeltaFromInTheMoneyProb = CND(
            CNDEV(InTheMoneyProb * Exp((b - r) * T)) - v * Sqr(T)
        )
    else:
        GDeltaFromInTheMoneyProb = -CND(
            CNDEV(InTheMoneyProb * Exp((b - r) * T)) + v * Sqr(T)
        )
    return GDeltaFromInTheMoneyProb


# MirrorDeltaStrike, delta neutral straddle strike in the BSM formula
def GDeltaMirrorStrike(S: float, T: float, b: float, v: float):
    GDeltaMirrorStrike = S * Exp((b + v ^ 2 / 2) * T)
    return GDeltaMirrorStrike


# MirrorProbabilityStrike, probability neutral straddle strike in the BSM formula
def GProbabilityMirrorStrike(S: float, T: float, b: float, v: float):
    GProbabilityMirrorStrike = S * Exp((b - v ^ 2 / 2) * T)
    return GProbabilityMirrorStrike


# MirrorDeltaStrike, general delta symmmetric strike in the BSM formula
def GDeltaMirrorCallPutStrike(S: float, X: float, T: float, b: float, v: float):
    GDeltaMirrorCallPutStrike = S ^ 2 / X * Exp((2 * b + v ^ 2) * T)
    return GDeltaMirrorCallPutStrike


# Gamma for the generalized Black and Scholes formula
def GGamma(S: float, X: float, T: float, r: float, b: float, v: float):
    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    GGamma = Exp((b - r) * T) * ND(d1) / (S * v * Sqr(T))
    return GGamma


# SaddleGamma for the generalized Black and Scholes formula
def GSaddleGamma(X: float, T: float, r: float, b: float, v: float):
    GSaddleGamma = Sqr(Exp(1) / np.pi) * Sqr((2 * b - r) / v ^ 2 + 1) / X
    return GSaddleGamma


# GammaP for the generalized Black and Scholes formula
def GGammaP(S: float, X: float, T: float, r: float, b: float, v: float):
    GGammaP = S * GGamma(S, X, T, r, b, v) / 100
    return GGammaP


# Delta for the generalized Black and Scholes formula
def GDelta(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    if CallPutFlag == "c":
        GDelta = Exp((b - r) * T) * CND(d1)
    else:
        GDelta = -Exp((b - r) * T) * CND(-d1)
    return GDelta


# StrikeDelta for the generalized Black and Scholes formula
def GStrikeDelta(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
):

    d2 = (Log(S / X) + (b - v ^ 2 / 2) * T) / (v * Sqr(T))
    if CallPutFlag == "c":
        GStrikeDelta = -Exp(-r * T) * CND(d2)
    else:
        GStrikeDelta = Exp(-r * T) * CND(-d2)
    return GStrikeDelta


# Elasticity for the generalized Black and Scholes formula
def GElasticity(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
):

    GElasticity = (
        GDelta(CallPutFlag, S, X, T, r, b, v)
        * S
        / GBlackScholes(CallPutFlag, S, X, T, r, b, v)
    )
    return GElasticity


# DgammaDvol/Zomma for the generalized Black and Scholes formula
def GDgammaDvol(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    GDgammaDvol = GGamma(S, X, T, r, b, v) * ((d1 * d2 - 1) / v)
    return GDgammaDvol


# DgammaPDvol for the generalized Black and Scholes formula
def GDgammaPDvol(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    GDgammaPDvol = S / 100 * GGamma(S, X, T, r, b, v) * ((d1 * d2 - 1) / v)
    return GDgammaPDvol


# DgammaDspot/Speed for the generalized Black and Scholes formula
def GDgammaDspot(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))

    GDgammaDspot = -GGamma(S, X, T, r, b, v) * (1 + d1 / (v * Sqr(T))) / S
    return GDgammaDspot


# DgammaPDspot/SpeedP for the generalized Black and Scholes formula
def GDgammaPDspot(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))

    GDgammaPDspot = -GGamma(S, X, T, r, b, v) * (d1) / (100 * v * Sqr(T))
    return GDgammaPDspot


# Risk Neutral Denisty for the generalized Black and Scholes formula
def GRiskNeutralDensity(S: float, X: float, T: float, r: float, b: float, v: float):

    d2 = (Log(S / X) + (b - v ^ 2 / 2) * T) / (v * Sqr(T))
    GRiskNeutralDensity = Exp(-r * T) * ND(d2) / (X * v * Sqr(T))
    return GRiskNeutralDensity


# Volatility estimate confidence interval
def GConfidenceIntervalVolatility(
    Alfa: float, n: int, VolatilityEstimate: float, UpperLower: str
):
    # UpperLower     ="L" gives the lower cofidence interval
    #               ="U" gives the upper cofidence interval
    # n: number of observations
    if UpperLower == "L":
        GConfidenceIntervalVolatility = VolatilityEstimate * Sqr(
            (n - 1) / (chi2.ppf(Alfa / 2, n - 1))
        )
    elif UpperLower == "U":
        GConfidenceIntervalVolatility = VolatilityEstimate * Sqr(
            (n - 1) / (chi2.ppf(1 - Alfa / 2, n - 1))
        )
    return GConfidenceIntervalVolatility


# Theta for the generalized Black and Scholes formula
def GTheta(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)

    if CallPutFlag == "c":
        GTheta = (
            -S * Exp((b - r) * T) * ND(d1) * v / (2 * Sqr(T))
            - (b - r) * S * Exp((b - r) * T) * CND(d1)
            - r * X * Exp(-r * T) * CND(d2)
        )
    elif CallPutFlag == "p":
        GTheta = (
            -S * Exp((b - r) * T) * ND(d1) * v / (2 * Sqr(T))
            + (b - r) * S * Exp((b - r) * T) * CND(-d1)
            + r * X * Exp(-r * T) * CND(-d2)
        )
    return GTheta


# Drift-less Theta for the generalized Black and Scholes formula
def GThetaDriftLess(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    GThetaDriftLess = -S * Exp((b - r) * T) * ND(d1) * v / (2 * Sqr(T))
    return GThetaDriftLess


# Variance-vega for the generalized Black and Scholes formula
def GVarianceVega(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    GVarianceVega = S * Exp((b - r) * T) * ND(d1) * Sqr(T) / (2 * v)
    return GVarianceVega


# Variance-vomma for the generalized Black and Scholes formula
def GVarianceVomma(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    GVarianceVomma = (
        S * Exp((b - r) * T) * Sqr(T) / (4 * v ^ 3) * ND(d1) * (d1 * d2 - 1)
    )
    return GVarianceVomma


# Variance-delta for the generalized Black and Scholes formula
def GVarianceDelta(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    GVarianceDelta = S * Exp((b - r) * T) * ND(d1) * (-d2) / (2 * v ^ 2)
    return GVarianceDelta


# Vega for the generalized Black and Scholes formula
def GVega(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    GVega = S * Exp((b - r) * T) * ND(d1) * Sqr(T)
    return GVega


# VegaP for the generalized Black and Scholes formula
def GVegaP(S: float, X: float, T: float, r: float, b: float, v: float):

    GVegaP = v / 10 * GVega(S, X, T, r, b, v)
    return GVegaP


# DdeltaDtime/Charm for the generalized Black and Scholes formula
def GDdeltaDtime(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)

    if CallPutFlag == "c":
        GDdeltaDtime = -Exp((b - r) * T) * (
            ND(d1) * (b / (v * Sqr(T)) - d2 / (2 * T)) + (b - r) * CND(d1)
        )
    elif CallPutFlag == "p":
        GDdeltaDtime = -Exp((b - r) * T) * (
            ND(d1) * (b / (v * Sqr(T)) - d2 / (2 * T)) - (b - r) * CND(-d1)
        )
    return GDdeltaDtime


# Profitt/Loss STD for the generalized Black and Scholes formula
def GProfitLossSTD(
    TypeFlag: str,
    CallPutFlag: str,
    S: float,
    X: float,
    T: float,
    r: float,
    b: float,
    v: float,
    NHedges: int,
):

    if TypeFlag == "a":  # in dollars
        GProfitLossSTD = Sqr(np.pi / 4) * GVega(S, X, T, r, b, v) * v / Sqr(NHedges)
    elif TypeFlag == "p":  # in percent
        GProfitLossSTD = (
            Sqr(np.pi / 4)
            * GVega(S, X, T, r, b, v)
            * v
            / Sqr(NHedges)
            / GBlackScholes(CallPutFlag, S, X, T, r, b, v)
        )
    return GProfitLossSTD


# DvegaDvol/Vomma for the generalized Black and Scholes formula
def GDvegaDvol(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    GDvegaDvol = GVega(S, X, T, r, b, v) * d1 * d2 / v
    return GDvegaDvol


# DvegaPDvol/VommaP for the generalized Black and Scholes formula
def GDvegaPDvol(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    GDvegaPDvol = GVegaP(S, X, T, r, b, v) * d1 * d2 / v
    return GDvegaPDvol


# DvegaDtime for the generalized Black and Scholes formula
def GDvegaDtime(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    GDvegaDtime = GVega(S, X, T, r, b, v) * (
        r - b + b * d1 / (v * Sqr(T)) - (1 + d1 * d2) / (2 * T)
    )
    return GDvegaDtime


# DVommaDVol for the generalized Black and Scholes formula
def GDvommaDvol(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    GDvommaDvol = (
        GDvegaDvol(S, X, T, r, b, v) * 1 / v * (d1 * d2 - d1 / d2 - d2 / d1 - 1)
    )
    return GDvommaDvol


# GGammaDtime for the generalized Black and Scholes formula
def GDgammaDtime(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    GDgammaDtime = GGamma(S, X, T, r, b, v) * (
        r - b + b * d1 / (v * Sqr(T)) + (1 - d1 * d2) / (2 * T)
    )
    return GDgammaDtime


# GGammaPDtime for the generalized Black and Scholes formula
def GDgammaPDtime(S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    GDgammaPDtime = GGammaP(S, X, T, r, b, v) * (
        r - b + b * d1 / (v * Sqr(T)) + (1 - d1 * d2) / (2 * T)
    )
    return GDgammaPDtime


# Vega for the generalized Black and Scholes formula
def GVegaLeverage(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
):

    GVegaLeverage = (
        GVega(S, X, T, r, b, v) * v / GBlackScholes(CallPutFlag, S, X, T, r, b, v)
    )
    return GVegaLeverage


# Rho for the generalized Black and Scholes formula for all options except futures
def GRho(CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    d2 = d1 - v * Sqr(T)
    if CallPutFlag == "c":
        GRho = T * X * Exp(-r * T) * CND(d2)
    elif CallPutFlag == "p":
        GRho = -T * X * Exp(-r * T) * CND(-d2)
    return GRho


# Rho for the generalized Black and Scholes formula for Futures option
def GRhoFO(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
):

    GRhoFO = -T * GBlackScholes(CallPutFlag, S, X, T, r, 0, v)
    return GRhoFO


# Rho2/Phi for the generalized Black and Scholes formula
def GPhi(CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    if CallPutFlag == "c":
        GPhi = -T * S * Exp((b - r) * T) * CND(d1)
    elif CallPutFlag == "p":
        GPhi = T * S * Exp((b - r) * T) * CND(-d1)
    return GPhi


# Carry rho sensitivity for the generalized Black and Scholes formula
def GCarry(
    CallPutFlag: str, S: float, X: float, T: float, r: float, b: float, v: float
):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T) / (v * Sqr(T))
    if CallPutFlag == "c":
        GCarry = T * S * Exp((b - r) * T) * CND(d1)
    elif CallPutFlag == "p":
        GCarry = -T * S * Exp((b - r) * T) * CND(-d1)
    return GCarry


class Array:
    def __init__(self, *args):
        self.array = list(args)

    def __call__(self, index):
        return self.array[index]


# Inverse cummulative normal distribution function
def CNDEV(U: float):

    A = Array(2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637)
    b = Array(-8.4735109309, 23.08336743743, -21.06224101826, 3.13082909833)
    c = Array(
        0.337475482272615,
        0.976169019091719,
        0.160797971491821,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        3.21767881767818e-05,
        2.888167364e-07,
        3.960315187e-07,
    )

    X = U - 0.5
    if abs(X) < 0.92:
        r = X * X
        r = (
            X
            * (((A(3) * r + A(2)) * r + A(1)) * r + A(0))
            / ((((b(3) * r + b(2)) * r + b(1)) * r + b(0)) * r + 1)
        )
        CNDEV = r
        return CNDEV

    r = U
    if X >= 0:
        r = 1 - U
    r = Log(-Log(r))
    r = c(0) + r * (
        c(1)
        + r
        * (
            c(2)
            + r * (c(3) + r + (c(4) + r * (c(5) + r * (c(6) + r * (c(7) + r * c(8))))))
        )
    )
    if X < 0:
        r = -r
    CNDEV = r
    return CNDEV


# The normal distribution function
def ND(X: float) -> float:
    ND = 1 / Sqr(2 * np.pi) * Exp(-X ^ 2 / 2)
    return ND


# Partial-time fixed strike lookback options
def PartialFixedLB(
    CallPutFlag: str,
    S: float,
    X: float,
    t1: float,
    T2: float,
    r: float,
    b: float,
    v: float,
):

    d1 = (Log(S / X) + (b + v ^ 2 / 2) * T2) / (v * Sqr(T2))
    d2 = d1 - v * Sqr(T2)
    e1 = ((b + v ^ 2 / 2) * (T2 - t1)) / (v * Sqr(T2 - t1))
    e2 = e1 - v * Sqr(T2 - t1)
    f1 = (Log(S / X) + (b + v ^ 2 / 2) * t1) / (v * Sqr(t1))
    f2 = f1 - v * Sqr(t1)
    if CallPutFlag == "c":
        PartialFixedLB = S * Exp((b - r) * T2) * CND(d1) - Exp(-r * T2) * X * CND(
            d2
        ) + S * Exp(-r * T2) * v ^ 2 / (2 * b) * (
            -(S / X)
            ^ (-2 * b / v ^ 2)
            * CBND(d1 - 2 * b * Sqr(T2) / v, -f1 + 2 * b * Sqr(t1) / v, -Sqr(t1 / T2))
            + Exp(b * T2) * CBND(e1, d1, Sqr(1 - t1 / T2))
        ) - S * Exp(
            (b - r) * T2
        ) * CBND(
            -e1, d1, -Sqr(1 - t1 / T2)
        ) - X * Exp(
            -r * T2
        ) * CBND(
            f2, -d2, -Sqr(t1 / T2)
        ) + Exp(
            -b * (T2 - t1)
        ) * (
            1 - v ^ 2 / (2 * b)
        ) * S * Exp(
            (b - r) * T2
        ) * CND(
            f1
        ) * CND(
            -e2
        )
    elif CallPutFlag == "p":
        PartialFixedLB = X * Exp(-r * T2) * CND(-d2) - S * Exp((b - r) * T2) * CND(
            -d1
        ) + S * Exp(-r * T2) * v ^ 2 / (2 * b) * (
            (S / X)
            ^ (-2 * b / v ^ 2)
            * CBND(-d1 + 2 * b * Sqr(T2) / v, f1 - 2 * b * Sqr(t1) / v, -Sqr(t1 / T2))
            - Exp(b * T2) * CBND(-e1, -d1, Sqr(1 - t1 / T2))
        ) + S * Exp(
            (b - r) * T2
        ) * CBND(
            e1, -d1, -Sqr(1 - t1 / T2)
        ) + X * Exp(
            -r * T2
        ) * CBND(
            -f2, d2, -Sqr(t1 / T2)
        ) - Exp(
            -b * (T2 - t1)
        ) * (
            1 - v ^ 2 / (2 * b)
        ) * S * Exp(
            (b - r) * T2
        ) * CND(
            -f1
        ) * CND(
            e2
        )
    return PartialFixedLB
