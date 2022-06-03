import re

import matplotlib.pyplot as plt

PROFIT_COLOR = "#002856"
LOSS_COLOR = "#f99e74"


class Option:
    def __init__(self, strike, premium, side="long", kind="call", **kwargs):
        """
        The __init__ function is called every time a new object is created.
        The values we pass to __init__ are stored in the object and can be accessed later

        :param self: Refer to the object itself
        :param strike: Define the strike price of the option
        :param premium: Set the price of the option
        :param side=&quot;long&quot;: Indicate whether the option is held long or short
        :param kind=&quot;call&quot;: Tell the class whether it is a call or put option
        :return: The object of the class
        :doc-author: Trelent
        """

        self.strike = strike
        self.premium = premium
        self.side = side
        self.kind = kind

    def get_profit(self, current_value):
        if self.side == "long":
            if self.kind == "call":
                if current_value < self.strike:
                    return -self.premium
                else:
                    return current_value - self.strike - self.premium
            elif self.kind == "put":
                if current_value > self.strike:
                    return -self.premium
                else:
                    return self.strike - current_value - self.premium
        elif self.side == "short":
            self.side = "long"
            profit = -self.get_profit(current_value)
            self.side = "short"
            return profit

    def plot(self, start, end, ax=None, figsize=(12, 10), **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        profits = [
            self.get_profit(current_value) for current_value in range(start, end)
        ]
        ax.plot(range(start, end), profits, **kwargs, alpha=0.3)
        ax.hlines(0, start, end, color="black")
        return ax

    def __repr__(self):
        return f"Option(strike={self.strike}, premium={self.premium}, side={self.side}, kind={self.kind})"


class ExoticOption(Option):
    def __init__(
        self,
        strike,
        premium=None,
        side="long",
        kind="call",
        condition=None,
        barrier=None,
    ):
        self.strike = strike
        self.premium = premium if premium is not None else 0
        self.side = side
        self.kind = kind
        if condition is not None:
            assert barrier is not None

        self.condition = self.parse_condition(condition)
        self.barrier = barrier
        self.price_barrier = self.get_price_barrier()

    def parse_condition(self, condition):
        if self.kind == "call":
            key = "c"
        else:
            key = "p"
        if re.search(r"[Uu][pP]", condition):
            key += "u"
        else:
            key += "d"
        if re.search(r"[iI][nN]", condition):
            key += "i"
        else:
            key += "o"
        return key

    def get_price_barrier(self):
        if "u" in self.condition:
            price_barrier = self.strike * self.barrier
        else:
            price_barrier = self.strike * self.barrier

        return price_barrier

    def get_profit(self, current_value):
        regular_profit = super().get_profit(current_value)
        if self.condition.endswith("di"):
            if current_value < self.price_barrier:
                return regular_profit
            else:
                return super().get_profit(self.strike)

        elif self.condition.endswith("ui"):
            if current_value > self.price_barrier:
                return regular_profit
            else:
                return super().get_profit(self.strike)

        elif self.condition.endswith("do"):
            if current_value > self.price_barrier:
                return regular_profit
            else:
                return super().get_profit(self.strike)

        elif self.condition.endswith("uo"):
            if current_value < self.price_barrier:
                return regular_profit
            else:
                return super().get_profit(self.strike)

    def __repr__(self):
        return f"ExoticOption(strike={self.strike}, premium={self.premium}, side={self.side}, kind={self.kind}, condition={self.condition}, barrier={self.barrier:.2%})"


class OptionsStrategy:
    def __init__(self):
        self.options = []
        self.premium = 0

    def get_profit(self, current_value):
        profit = 0
        for option in self.options:
            profit += option.get_profit(current_value)
        return profit

    def add(self, *args, **kwargs):
        """
        The add function is called every time a new option is added.

        :param self: Refer to the object itself
        :param strike: Define the strike price of the option
        :param premium: Set the price of the option
        :param side=&quot;long&quot;: Indicate whether the option is held long or short
        :param kind=&quot;call&quot;: Tell the class whether it is a call or put option
        :return: The object of the class
        :doc-author: Trelent
        """
        if "condition" in kwargs:
            option = ExoticOption(*args, **kwargs)
        else:
            option = Option(*args, **kwargs)
        self.premium += option.premium
        self.options.append(option)
        return self

    def remove(self, *args, **kwargs):
        option = Option(*args, **kwargs)
        for opt in self.options:
            if opt == option:
                self.premium -= option.premium
                self.options.remove(opt)
                return self

    def clear(self):
        self.options = []
        self.premium = 0

    def __repr__(self):
        options = "\n".join([str(i) for i in self.options])
        return f"OptionsStrategy([{options}])"

    def plot(self, start=None, end=None, fig_size=(12, 12), **kwargs):
        if start is None:
            start = int(min([option.strike for option in self.options]) / 2)
        if end is None:
            end = max([option.strike for option in self.options])
            end = int(end + end / 2)
        fig, ax = plt.subplots(figsize=fig_size)
        for option in self.options:
            option.plot(
                start,
                end,
                ax=ax,
                linestyle="dashed",
                label=str(option),
                color=LOSS_COLOR,
                **kwargs,
            )

        ax.plot(
            range(start, end),
            [self.get_profit(current_value) for current_value in range(start, end)],
            label="Strategy",
            color=PROFIT_COLOR,
            **kwargs,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        # Change ticks font size
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

        ax.legend()
        return ax


class BSOptionsStrategy(OptionsStrategy):
    def __init__(
        self,
        ticker=None,
        ticker_df=None,
        start_date_for_data="29/03/2021",
        r=0.04,
        days=365,
    ):
        from black_and_scholes_alan_genz import BlackAndScholes

        super().__init__()
        self.ticker = ticker
        if ticker_df is None:
            self.b_and_s = BlackAndScholes(
                ticker, start_date_for_data=start_date_for_data, r=r, days=days
            )
        else:
            self.b_and_s = BlackAndScholes(ticker, ticker_df=ticker_df, r=r, days=days)
        self.ticker_df = self.b_and_s.ticker_df
        self.premium = 0
        self.last_price = self.b_and_s.last_price

    def add(self, *args, **kwargs):
        """
        The add function is called every time a new option is added.

        :param self: Refer to the object itself
        :param strike: Define the strike price of the option
        :param premium: Set the price of the option
        :param side=&quot;long&quot;: Indicate whether the option is held long or short
        :param kind=&quot;call&quot;: Tell the class whether it is a call or put option
        :return: The object of the class
        :doc-author: Trelent
        """
        if "condition" in kwargs:
            if kwargs["condition"] == "vanilla":
                self.add_vanilla(*args, **kwargs)
            else:
                self.add_exotic(*args, **kwargs)
        else:
            self.add_vanilla(*args, **kwargs)
        return self

    def add_exotic(self, *args, **kwargs):
        if "strike" not in kwargs:
            kwargs["strike"] = self.last_price
        dummy_option = ExoticOption(*args, **kwargs)
        premium = self.b_and_s.price_exotic(dummy_option)
        kwargs["premium"] = premium
        option = ExoticOption(*args, **kwargs)
        self.premium += option.premium
        self.options.append(option)
        return self

    def add_vanilla(self, *args, **kwargs):
        if "strike" not in kwargs and "premium" not in kwargs:
            kwargs["strike"] = self.b_and_s.last_price
        if "strike" not in kwargs and "premium" in kwargs:
            kwargs["strike"] = self.get_strike(kwargs)
        if "premium" not in kwargs:
            kwargs["premium"] = self.get_premium(kwargs)
        option = Option(*args, **kwargs)
        self.premium += option.premium
        self.options.append(option)
        return self

    def get_premium(self, kwargs):
        result = self.b_and_s.price(strike=kwargs["strike"])
        if kwargs["kind"] == "call":
            return result["call"]
        else:
            return result["put"]

    def get_strike(self, kwargs):
        if kwargs["kind"] == "call":
            return self.b_and_s.look_for_strike(call=kwargs["premium"])
        else:
            return self.b_and_s.look_for_strike(put=kwargs["premium"])

    def get_price_series(self):
        return self.b_and_s.price_series
