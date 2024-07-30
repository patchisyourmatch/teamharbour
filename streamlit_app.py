#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
import streamlit as st

# Functions

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

def calculate_var(returns, confidence_level=0.95):
    return returns.mean() - norm.ppf(confidence_level) * returns.std()

def calculate_cvar(returns, confidence_level=0.95):
    var = calculate_var(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    return cvar

# Team Harbour Landing Page with Loading Progress Bar
st.title("Welcome to Team Harbour's Portfolio Optimization Tool")
st.write("Optimize your portfolio for minimal risk with our advanced analytics tool.")

progress_bar = st.progress(0)
status_text = st.empty()

for percent_complete in range(100):
    time.sleep(0.01)
    progress_bar.progress(percent_complete + 1)
    status_text.text(f'Loading... {percent_complete + 1}%')

if st.button('Start'):
    st.session_state.page = 'main'

if 'page' not in st.session_state:
    st.session_state.page = 'landing'

if st.session_state.page == 'landing':
    st.stop()

tickers = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, MSFT, GOOGL)")

if tickers:
    tickers = [ticker.strip().upper() for ticker in tickers.split(',')]

    asset_data = yf.download(
        tickers=' '.join(tickers), 
        start='2020-01-01',
        end='2024-07-26',
        progress=False  # Disable progress bar
    )['Adj Close']

    if asset_data.isnull().values.any():
        st.error("One or more tickers are invalid or do not have enough data.")
    else:
        daily_returns = asset_data.pct_change().dropna()
        mean_rets = daily_returns.mean() * 252
        covar = daily_returns.cov() * 252

        st.write("Covariance Matrix:")
        st.dataframe(covar)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 0.35) for asset in range(len(tickers)))
        init_guess = len(tickers) * [1. / len(tickers),]

        opt_results = minimize(portfolio_volatility, init_guess, args=(covar,), method='SLSQP', bounds=bounds, constraints=constraints)

        min_vol_weights = opt_results.x
        min_vol = portfolio_volatility(min_vol_weights, covar)
        expected_return = np.dot(min_vol_weights, mean_rets)

        st.markdown(f"""
        ### Portfolio Optimization Results

        | **Metric**                                | **Value**               |
        |-------------------------------------------|-------------------------|
        | **1. Minimum volatility**                 | **{min_vol:.12f}**      |
        | **2. Expected return of minimum volatility portfolio** | **{expected_return:.12f}** |
        """)

        st.markdown("### Optimal Weights for Minimal Volatility")
        weights_table = "| **Stock** | **Weight**   |\n|-----------|--------------|\n"
        for weight, stock in zip(min_vol_weights, tickers):
            weights_table += f"| **{stock}** | **{round(weight * 100, 2)}%** |\n"
        st.markdown(weights_table)

        st.markdown("### Portfolio Composition")
        fig, ax = plt.subplots()
        ax.pie(min_vol_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        var_95 = calculate_var(daily_returns, 0.95)
        cvar_95 = calculate_cvar(daily_returns, 0.95)

        st.write(f"VaR (95%): {var_95}")
        st.write(f"CVaR (95%): {cvar_95}")

        st.markdown("### Efficient Frontier")

        frontier_progress_bar = st.progress(0)
        frontier_status_text = st.empty()

        n = 50000
        port_weights = np.zeros((n, len(tickers)))
        port_volatility = np.zeros(n)
        port_return = np.zeros(n)
        port_sr = np.zeros(n)

        for i in range(n):
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)
            port_weights[i, :] = weights 

            port_return[i] = np.sum(mean_rets * weights)
            port_volatility[i] = portfolio_volatility(weights, covar)
            port_sr[i] = port_return[i] / port_volatility[i]
            
            if i % (n // 100) == 0:
                percent_complete = int((i / n) * 100)
                frontier_progress_bar.progress(percent_complete + 1)
                frontier_status_text.text(f'Loading Efficient Frontier & Portfolio Performance... {percent_complete + 1}%')

        frontier_progress_bar.empty()
        frontier_status_text.empty()

        plt.figure(figsize=(20, 15))
        plt.scatter(port_volatility, port_return, c=port_sr, cmap='plasma')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.title('Efficient Frontier (Bullet Plot)')
        plt.scatter(min_vol, expected_return, c='blue', s=150, edgecolors='red', marker='o', label='Min Volatility Portfolio')
        plt.legend()
        st.pyplot(plt)

        st.markdown("### Portfolio Performance")

        def plot_performance(start_date, end_date, title):
            price_data = yf.download(
                tickers=' '.join(tickers) + ' SPY', 
                start=start_date,
                end=end_date,
                progress=False  # Disable progress bar
            )['Adj Close']

            ret_data = price_data.pct_change().dropna()
            weighted_returns = min_vol_weights * ret_data[tickers]
            port_ret = weighted_returns.sum(axis=1)
            cumulative_ret = (port_ret + 1).cumprod() - 1

            spy_ret = (ret_data['SPY'] + 1).cumprod() - 1

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(cumulative_ret, label='Portfolio')
            ax.plot(spy_ret, label='SPY')
            ax.set_xlabel('Date')
            ax.set_ylabel("Cumulative Returns")
            ax.set_title(title)
            ax.legend()
            st.pyplot(fig)

        plot_performance('2024-01-01', '2024-07-25', 'Portfolio vs SPY (2024-01-01 to 2024-07-25)')
        plot_performance('2023-07-27', '2023-10-27', 'Portfolio vs SPY (2023-07-27 to 2023-10-27)')
        plot_performance('2022-01-04', '2022-10-13', 'Portfolio vs SPY (2022-01-04 to 2022-10-13)')
        plot_performance('2020-02-19', '2020-03-23', 'Portfolio vs SPY (2020-02-19 to 2020-03-23)')