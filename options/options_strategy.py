import matplotlib.pyplot as plt

from black_and_scholes import BlackAndScholes


class Option:
    def __init__(self, strike, premium, side="long", kind="call"):
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


class OptionsStrategy:
    def __init__(self):
        self.options = []

    def get_profit(self, current_value):
        profit = 0
        for option in self.options:
            profit += option.get_profit(current_value)
        return profit

    def add(self, *args, **kwargs):
        self.options.append(Option(*args, **kwargs))
        return self

    def remove(self, *args, **kwargs):
        option = Option(*args, **kwargs)
        for opt in self.options:
            if opt == option:
                self.options.remove(opt)
                return self

    def clear(self):
        self.options = []

    def __repr__(self):
        return f"OptionsStrategy({self.options})"

    def plot(self, start=None, end=None, fig_size=(12, 12), **kwargs):
        if start is None:
            start = int(min([option.strike for option in self.options]) / 2)
        if end is None:
            end = max([option.strike for option in self.options])
            end = int(end + end / 2)
        fig, ax = plt.subplots(figsize=fig_size)
        for option in self.options:
            option.plot(
                start, end, ax=ax, linestyle="dashed", label=str(option), **kwargs
            )

        ax.plot(
            range(start, end),
            [self.get_profit(current_value) for current_value in range(start, end)],
            label="Strategy",
            **kwargs,
        )
        ax.legend()
        return ax


class BSOptionsStrategy(OptionsStrategy):
    def __init__(self, ticker, start_date_for_data="29/03/2021", r=0.01988, days=365):
        super().__init__()
        self.ticker = ticker
        self.b_and_s = BlackAndScholes(
            ticker, start_date_for_data=start_date_for_data, r=r, days=days
        )

    def add(self, *args, **kwargs):
        if "strike" not in kwargs:
            kwargs["strike"] = self.b_and_s.last_price
        kwargs["premium"] = self.get_premium(kwargs)
        self.options.append(Option(*args, **kwargs))
        return self

    def get_premium(self, kwargs):
        result = self.b_and_s.price(strike=kwargs["strike"])
        if kwargs["kind"] == "call":
            return result["call"]
        else:
            return result["put"]

    def get_price_series(self):
        return self.b_and_s.price_series
