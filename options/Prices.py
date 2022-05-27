import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import objective_functions
import investpy
from datetime import datetime
import plotly.express as px


def get_df(ticker, start, end=None):
    end = end if end else datetime.today().strftime("%d/%m/%Y")
    result = investpy.search_quotes(text=ticker, n_results=1)
    df = result.retrieve_historical_data(from_date=start, to_date=end)
    return df


class TickerAnalyst:
    def __init__(self, tickers, benchmark=None, forex=[], start="01/01/2018", end=None):
        self.start = start
        self.end = end if end else datetime.today().strftime("%d/%m/%Y")
        self.tickers = tickers
        self.benchmark = benchmark
        self.forex = forex
        self.df = self.get_df()
        self.get_price_dfs()
        self.get_returns()
        self.betas = self.get_betas()
        self.weights = self.get_weights()

    def get_df(self):
        tickers_to_search = self.tickers + [self.benchmark] + self.forex
        dfs = []
        for ticker in tickers_to_search:
            if ticker is not None:
                print(f"Searching for {ticker}")
                result = investpy.search_quotes(text=ticker, n_results=1)
                print("Found: ", result)
                df = result.retrieve_historical_data(
                    from_date=self.start, to_date=self.end
                )
                dfs.append(df)

        df = pd.concat([df[["Close"]] for df in dfs], axis=1)
        df = df.dropna()
        df.columns = filter(
            lambda x: x is not None,
            tickers_to_search,
        )
        return df

    def get_price_dfs(self):
        self.daily_df = self.df.fillna(method="ffill").copy()
        self.weekly_df = self.df.resample("w").last().copy()
        self.monthly_df = self.df.resample("M").last().copy()
        self.yearly_df = self.df.resample("Y").last().copy()

    def get_returns(self):
        self.daily_returns = self.daily_df.pct_change().dropna()
        self.weekly_returns = self.weekly_df.pct_change().dropna()
        self.monthly_returns = self.monthly_df.pct_change().dropna()
        self.yearly_returns = self.yearly_df.pct_change().dropna()

    def _get_betas(self, stocks_df, benchmark_df):
        betas = {}
        benchmark_var = np.var(benchmark_df)
        for ticker in self.tickers:
            try:
                stock_series = stocks_df[ticker].dropna()
                benchmark_series = benchmark_df.iloc[-len(stock_series) :]
                cov = np.cov(stock_series, benchmark_series)[0, 1]
                beta = cov / benchmark_var
                betas[ticker] = beta
            except:
                print("Error with ", ticker)
                continue
        betas = pd.Series(betas)
        betas.name = "Betas"
        return betas

    def get_betas(self, timeframe="d"):
        if timeframe == "d":
            return self._get_betas(
                self.daily_returns, self.daily_returns[self.benchmark]
            )
        if timeframe == "w":
            return self._get_betas(
                self.weekly_returns, self.weekly_returns[self.benchmark]
            )
        if timeframe == "m":
            return self._get_betas(
                self.monthly_returns, self.monthly_returns[self.benchmark]
            )
        if timeframe == "y":
            return self._get_betas(
                self.yearly_returns, self.yearly_returns[self.benchmark]
            )
        raise ValueError("Invalid timeframe")

    def get_corr(self):
        return (
            self.daily_df.corr()
            .style.background_gradient(cmap="Blues_r")
            .set_properties(**{"font-size": "12pt", "text-align": "center"})
        )

    def get_weights(self, method="max_sharpe", gamma=0.1):
        df = self.daily_df[self.tickers]
        if method == "max_sharpe":
            mu = mean_historical_return(df)
            S = CovarianceShrinkage(df).ledoit_wolf()
            ef = EfficientFrontier(mu, S)
            # ef.add_objective(objective_functions.L2_reg, gamma=gamma)
            # ef.add_constraint(lambda x: x >= 0.05)
            # ef.add_constraint(lambda x: x[:-1] <= 0.25)
            # ef.add_constraint(lambda x: sum(x) == 1)
            # ef.add_constraint(lambda x: x[-1] >= 0.4)
            # weights = ef.max_sharpe()
            weights = ef.min_volatility()

        if method == "min_vol":
            mu = mean_historical_return(self.daily_df)
            S = CovarianceShrinkage(self.daily_df).ledoit_wolf()
            ef = EfficientFrontier(mu, S)
            weights = ef.min_volatility()
        df = pd.DataFrame(pd.Series(weights))
        df.columns = ["Weights"]
        df["Tickers"] = df.index
        return df

    def plot_pie(self, graph_filename, weight_method=None):
        """This method plots a pie chart of the weights with Blues colors

        Args:
            weight_method (str, optional): Defaults to "max_sharpe".
        """
        if weight_method is not None:
            df = self.get_weights(weight_method)
        else:
            df = self.weights
        fig = px.pie(df, values="Weights", names="Tickers", title="Weights")
        fig.update_traces(textposition="outside", hole=0.25, textinfo="percent+label")
        fig.write_image(graph_filename)
        fig.show()

    def get_corr_graph(
        self,
        filename=None,
        diagonal="kde",
        figsize=(12, 12),
        marker=".",
        alpha=0.2,
        **kwargs,
    ):
        plot = pd.plotting.scatter_matrix(
            self.daily_returns,
            diagonal=diagonal,
            figsize=figsize,
            marker=marker,
            alpha=alpha,
            **kwargs,
        )
        if filename:
            plt.savefig(filename)
        return plot

    def main(self, filename=None):
        self.betas = self.get_betas()
        self.weights = self.get_weights()
        self.to_excel(filename)

    def to_excel_light(self, filename):
        with pd.ExcelWriter(
            filename,
            datetime_format="YYYY-MM-DD",
            date_format="YYYY-MM-DD",
            engine="xlsxwriter",
        ) as excel_book:
            self.daily_df.to_excel(excel_book, sheet_name="Prices")
            self.daily_returns.to_excel(
                excel_book,
                sheet_name="Prices",
                startrow=1,
                startcol=len(self.tickers) + 3,
                index=False,
            )

    def to_excel(self, filename):
        with pd.ExcelWriter(
            filename,
            datetime_format="YYYY-MM-DD",
            date_format="YYYY-MM-DD",
            engine="xlsxwriter",
        ) as excel_book:
            self.daily_df.to_excel(excel_book, sheet_name="Prices")
            self.daily_returns.to_excel(
                excel_book,
                sheet_name="Prices",
                startrow=1,
                startcol=len(self.tickers) + 3,
                index=False,
            )
            self.get_corr().to_excel(excel_book, sheet_name="Correlation Matrix")
            self.betas.to_excel(excel_book, sheet_name="Beta")
            self.weights.to_excel(excel_book, sheet_name="Portafolio")

            # corr_worksheet = excel_book.sheets["Correlation Matrix"]
            graph_filename = filename.split(".")[1]
            # graph_filename1 = graph_filename + ".png"
            graph_filename2 = graph_filename + "2" + ".png"
            # corr_figure = self.get_corr_graph(filename=graph_filename1)
            # corr_worksheet.insert_image(f"A{len(self.tickers) + 3}", graph_filename1)

            self.plot_pie(graph_filename2)
            portafolio_worksheet = excel_book.sheets["Portafolio"]
            portafolio_worksheet.insert_image(
                f"A{len(self.tickers) + 3}", graph_filename2
            )
