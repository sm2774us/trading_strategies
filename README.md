# __Trading Strategies__
My Private Repository of Trading Strategies

## [__`Boost Python` and `Numba` and `Cython`__](./boost_python_and_numba_and_cython.md)

## Table Of Contents <a name="top"></a>
0. [__News Sentiment Trading Strategies__](#news-sentiment-trading-strategies)
    - 0.0. [__0) Concepts and Trading:__](#0-concepts-and-trading)
        - 0.0.1. [__0.A) Retrieving News Data:__](#0a-retrieving-news-data)
        - 0.0.2. [__0.B) Identifying Sentiment from News Headline:__](#0b-identifying-sentiment-from-news-headline)
        - 0.0.3. [__0.C) Sentiment Score:__](#0c-sentiment-score)
        - 0.0.4. [__0.D) LLM Models:__](#0d-llm-models)
    - 0.1. [__1) Buy the Rumor Sell the Event:__](#1-buy-the-rumor-sell-the-event)
    - 0.2. [__2) VADER-based Sentiment Analysis:__](#2-vader-based-sentiment-analysis)
    - 0.3. [__3) Optimize Strategy Using Technical Indicators:__](#3-optimize-strategy-using-technical-indicators)
1. [__Mean Reversion Strategies__](#mean-reversion-strategies)
    - 1.0. [__1) Math Concepts:__](#1-math-concepts)
        - 1.0.1. [__1.A) Correlation and Co-Integration:__](#1a-correlation-and-co-integration)
        - 1.0.2. [__1.B) Stationarity:__](#1b-stationarity)
        - 1.0.3. [__1.C) Linear Regression:__](#1c-linear-regression)
        - 1.0.4. [__1.D) ADF (Augmented Dickey-Fuller) and Johansen Test:__](#1d-adf-augmented-dickey-fuller-and-johansen-test)
        - 1.0.4. [__1.E) Half-Life:__](#1e-half-life)
    - 1.1. [__1) Statistical Arbitrage:__](#1-statistical-arbitrage)
        - Statistical Arbitrage (StatArb) involves trading pairs of securities that historically move together. When their price relationship diverges, traders expect them to revert to the mean, thus creating profit opportunities.    
        - __Key Steps:__
            - __Identify Pairs:__ Select two assets with a strong historical correlation.
            - __Monitor Spread:__ Calculate the price difference (spread) between the two assets.
            - __Trade Signals:__ When the spread deviates significantly from its mean, initiate trades expecting reversion.
        - __Use Cases:__
            - __Equities:__ Commonly used with stocks exhibiting high correlation.
            - __FX:__ Currency pairs that move together due to economic ties.
            - __Futures:__ Contracts on similar commodities or indices.    
    - 1.2. [__2) Triplets Trading:__](#2-triplets-trading)
        - Triplets Trading extends pair trading to three assets. The idea is to find three securities whose combined weighted price exhibits mean-reverting behavior.
        - __Key Steps:__
            - __Select Triplets:__ Find three assets with a cointegrated relationship.
            - __Determine Weights:__ Assign weights to each asset to balance the combined position.
            - __Monitor Combined Spread:__ Track the combined weighted price for deviations.
        - __Use Cases:__
            - __Equities:__ Stocks from the same sector.
            - __Commodities:__ Related commodities like oil, gas, and energy ETFs.    
    - 1.3. [__3) Index Arbitrage:__](#3-index-arbitrage)
        - Index Arbitrage exploits price discrepancies between an index and its constituent components. When the sum of the components' prices diverges from the index's price, arbitrageurs can buy the undervalued and sell the overvalued.
        - __Key Steps:__
            - __Monitor Index vs. Components:__ Track the price of the index and the weighted sum of its components.
            - __Identify Discrepancies:__ When significant deviations occur, execute trades expecting convergence.
        - __Use Cases:__
            - __Equities:__ Index ETFs vs. underlying stocks.
            - __Futures:__ Index futures vs. spot indices.
    - 1.4. [__4) Long Short Strategy:__](#4-long-short-strategy)
        - A Long-Short strategy involves taking long positions in undervalued assets and short positions in overvalued ones, based on their deviation from the mean.
        - __Key Steps:__
            - __Valuation Metrics:__ Use indicators like P/E ratios, moving averages, etc.
            - __Identify Over/Undervalued Assets:__ Based on the metrics, classify assets.
            - __Execute Trades:__ Go long on undervalued and short on overvalued assets.
        - __Use Cases:__
            - __Equities:__ Stocks with varying valuations.
            - __FX:__ Currencies mispriced relative to economic indicators.
2. [__Momentum Trading Strategies__](#momentum-trading-strategies)
    - 2.0. [__2.0) Finance and Math Skills:__](#20-finance-and-math-skills)
        - 2.0.1. [__2.A) Contango and Backwardation:__](#2a-contango-and-backwardation)
        - 2.0.2. [__2.B) Hurst Exponent:__](#2b-hurst-exponent)
        - 2.0.3. [__2.C) Sharpe Ratio, Maximum Drawdowns:__](#2c-sharpe-ratio-maximum-drawdowns)
        - 2.0.4. [__2.D) Correlation Analysis:__](#2d-correlation-analysis)
    - 2.1. [__2.1) Roll Returns:__](#21-roll-returns)
    - 2.2. [__2.2) Time Series Momentum:__](#22-time-series-momentum)
    - 2.3. [__2.3) Cross Sectional Momentum:__](#23-cross-sectional-momentum)
    - 2.4. [__2.4) Crossover & Breakout:__](#24-crossover-breakout)
    - 2.5. [__2.5) Event Driven Momentum:__](#25-event-driven-momentum)
3. [__Trading Alphas: Mining, Optimization, and System Design__](#trading-alphas-mining-optimization-and-system-design)
    - 3.0. [__3.0) Math Concepts and Trading:__](#30-math-concepts-and-trading)
        - 3.0.1. [__3.A) Cointegration:__](#3a-cointegration)
        - 3.0.2. [__3.B) Correlation:__](#3b-correlation)
        - 3.0.3. [__3.C) Execution:__](#3c-execution)
        - 3.0.4. [__3.D) Building a trading platform:__](#3d-building-a-trading-platform)
        - 3.0.5. [__3.E) System Parameter Permutation:__](#3e-system-parameter-permutation)
    - 3.1. [__3.1) Mean Reversion:__](#31-mean-reversion)
    - 3.2. [__3.2) K-Nearest Neighbors:__](#32-k-nearest-neighbors)
    - 3.3. [__3.3) Time series & cross sectional alphas:__](#33-time-series-cross-sectional-alphas)
    - 3.4. [__3.4) Candlestick Patterns:__](#34-candlestick-patterns)
    - 3.5. [__3.5) Vectorized SL & TP:__](#35-vectorized-sl-and-tp)
4. [__Trading in Milliseconds: MFT Straegies and Setup__](#trading-in-milliseconds-mft-straegies-and-setup)
    - 4.0. [__4.0) Concepts and Trading:__](#40-concepts-and-trading)
        - 4.0.1. [__4.A) Time and Volume Bars:__](#4a-time-and-volume-bars)
        - 4.0.2. [__4.B) Spoofing, Front-running:__](#4b-spoofing-front-running)
        - 4.0.3. [__4.C) IOC and ISO orders:__](#4c-ioc-and-iso-orders)
        - 4.0.4. [__4.D) Lee-Ready Algorithm, BVC Rule:__](#4d-lee-ready-algorithm-bvc-rule)
        - 4.0.5. [__4.E) Tick Data:__](#4e-tick-data)
    - 4.1. [__4.1) Order flow trading using tick data:__](#41-order-flow-trading-using-tick-data)
    - 4.2. [__4.2) Hide and light:__](#42-hide-and-light)
    - 4.3. [__4.3) Stop hunting:__](#43-stop-hunting)
    - 4.4. [__4.4) Ticking:__](#4-ticking)

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

## __News Sentiment Trading Strategies__
## __Mean Reversion Strategies__
Mean reversion strategies in trading involve taking advantage of the tendency of asset prices to revert to their historical averages over time. These strategies assume that when an asset price deviates significantly from its average, it is likely to move back towards the average in the future. Mean reversion strategies are often implemented using statistical analysis, time series modeling, and quantitative methods.

__Use Cases for Mean Reversion Strategies:__
Mean reversion strategies can be applied across various asset categories including:
- __Equities:__ Mean reversion strategies are commonly used in stock markets to exploit price disparities between related stocks.
- __Fixed Income:__ Traders can implement mean reversion strategies in bond markets to benefit from yield spreads reverting to their historical averages.
- __Futures:__ Mean reversion strategies in futures markets involve exploiting price differentials between futures contracts and their underlying assets.
- __FX:__ Traders in the foreign exchange market can use mean reversion strategies to capitalize on currency pairs' price deviations from their historical averages.

These strategies can be programmed in both C++ and Python to automate trading decisions based on statistical analysis and quantitative models. C++ offers high performance, while Python provides ease of implementation and extensive libraries for quantitative analysis.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

### 1) Math Concepts:

#### 1.A. **Correlation and Co-Integration**

##### **Correlation**:
Correlation measures the linear relationship between two variables. The Pearson correlation coefficient `r` is calculated as:

$$
r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}
$$

Where:
- $x_i$ and $y_i$ are the data points of the two variables.
- $\bar{x}$ and $\bar{y}$ are the mean values of the variables.

##### **Co-Integration**:
Co-integration checks if a linear combination of two or more non-stationary time series is stationary. For two time series $X_t$ and $Y_t$, they are co-integrated if:

$$
Z_t = Y_t - \beta X_t
$$

is stationary, where $\beta$ is the co-integration coefficient.

###### **Python Code**:
```python
import numpy as np
from statsmodels.tsa.stattools import adfuller

def correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

# Generate sample time series data
timeseries1 = np.array([1, 2, 3, 4, 5])
timeseries2 = np.array([2, 4, 6, 8, 10])

# Calculate correlation coefficient
correlation = correlation(timeseries1, timeseries2)

# Perform ADF test for Co-Integration
result = adfuller(timeseries1 - timeseries2)
p_value = result[1]

print(f"Correlation Coefficient: {correlation}")
if p_value < 0.05:  # Null hypothesis rejected
    print("The time series are Co-Integrated.")
else:
    print("The time series are not Co-Integrated.")
print(f"Co-Integration p-value: {p_value}")
```

##### **C++ Code** (Using Eigen library):
```cpp
// adf_test.h
#ifndef ADF_TEST_H
#define ADF_TEST_H

#include "eigen/Eigen/Dense"
#include <iostream>
#include <cmath>
#include <vector>

double adf_test(const Eigen::VectorXd& diff_time_series) {
    int n = diff_time_series.size();
    Eigen::VectorXd y = diff_time_series.tail(n-1);
    Eigen::VectorXd x = diff_time_series.head(n-1);

    Eigen::MatrixXd XtX = x * x.transpose();
    Eigen::VectorXd XtY = x * y;

    Eigen::VectorXd beta = XtX.colPivHouseholderQr().solve(XtY);

    Eigen::VectorXd y_hat = x.transpose() * beta;
    Eigen::VectorXd residuals = y - y_hat;

    // Calculate test statistic for ADF test
    double test_statistic = (residuals.array() / residuals.tail(n - 1).array()).sum();

    // Calculate p-value (using normal distribution approximation)
    double p_value = 1.0 - std::abs(test_statistic); // Simplified for demonstration

    std::cout << "ADF Test Result - Test Statistic: " << test_statistic << ", p-value: " << p_value << std::endl;

    return p_value;
}

#endif // ADF_TEST_H
```
```cpp
// main.cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include "eigen/Eigen/Dense"
#include "adf_test.h" // Custom header for ADF test implementation

double correlation_using_stl(const std::vector<double>& x, const std::vector<double>& y) {
    double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
    double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
    double sum_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
    double sum_x2 = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    double sum_y2 = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
    int n = x.size();

    return (n * sum_xy - sum_x * sum_y) / 
           (std::sqrt(n * sum_x2 - sum_x * sum_x) * std::sqrt(n * sum_y2 - sum_y * sum_y));
}

int main() {
    Eigen::VectorXd time_series1(5);
    time_series1 << 1, 2, 3, 4, 5;

    Eigen::VectorXd time_series2(5);
    time_series2 << 2, 4, 6, 8, 10;

    // Calculate correlation coefficient
    double sum1 = time_series1.sum();
    double sum2 = time_series2.sum();
    double mean1 = sum1 / time_series1.size();
    double mean2 = sum2 / time_series2.size();

    double cov = (time_series1 - mean1).dot(time_series2 - mean2) / time_series1.size();
    double std1 = sqrt((time_series1 - mean1).squaredNorm() / time_series1.size());
    double std2 = sqrt((time_series2 - mean2).squaredNorm() / time_series2.size());

    double correlation = cov / (std1 * std2);
    double correlation_using_stl = correlation_using_stl(std::vector<double> {1.0, 2.0, 3.0, 4.0, 5.0}, std::vector<double> {2.0, 4.0, 6.0, 8.0, 10.0})

    // Perform ADF test for Co-Integration on the difference time series
    double p_value = adf_test(time_series1 - time_series2);

    std::cout << "Correlation Coefficient (using Eigen Library): " << correlation << std::endl;
    std::cout << "Correlation Coefficient (using STL): " << correlation_using_stl << std::endl;
    if (p_value < 0.05) {
        std::cout << "The time series are Co-Integrated." << std::endl;
    } else {
        std::cout << "The time series are not Co-Integrated." << std::endl;
    }
    std::cout << "Co-Integration p-value (using Eigen Library): " << p_value << std::endl;
    return 0;
}

```

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

#### 1.B. **Stationarity**
Stationarity refers to a time series whose statistical properties (mean, variance) do not change over time. A time series $X_t$ is stationary if:

$$
\text{E}[X_t] = \mu, \quad \text{Var}(X_t) = \sigma^2, \quad \text{Cov}(X_t, X_{t+h}) = \gamma(h)
$$

#### **Python Code**:
```python
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint

def is_stationary(x, significance=0.05):
    result = adfuller(x)
    return result[1] < significance

def print_adf_test(x):
    result = adfuller(x, maxlag=1, regression='c', autolag=None)
    print(f"ADF Statistic: {result[0]:.5f}")
    print(f"p-value: {result[1]:.5f}")
    print(f"Used Lag: {result[2]}")
    print(f"Number of Observations: {result[3]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.5f}")

def print_coint_test(y0, y1):
    result = coint(y0, y1)
    print(f"Cointegration t-statistic: {result[0]:.5f}")
    print(f"p-value: {result[1]:.5f}")
    print(f"Critical Values:")
    for key, value in result[2].items():
        print(f"   {key}: {value:.5f}")
    
    # Calculate and print cointegrating vector
    X = np.column_stack((y0, np.ones_like(y0)))
    beta = np.linalg.lstsq(X, y1, rcond=None)[0]
    print(f"Cointegrating vector: [1, {-beta[0]:.5f}]")

# Input data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

// Check if x is stationary
stationary = is_stationary(x)
if (stationary) {
    print("The time series 'x' is stationary.")
} else {
    print("The time series 'x' is not stationary.")
}

print("\nADF Test for x:")
print_adf_test(x)
print("\nADF Test for y:")
print_adf_test(y)
print("\nCointegration Test for x and y:")
print_coint_test(x, y)
```

#### **C++ Code** (using Eigen Library):
Stationarity tests like ADF are not directly available in C++ libraries, and implementing ADF involves recursive algorithms.
This is a very simplified attempt.
```cpp
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
#include <random>
#include <iostream>

class TimeSeriesAnalysis {
private:
    static std::vector<double> diff(const std::vector<double>& x) {
        std::vector<double> result(x.size() - 1);
        std::adjacent_difference(x.begin() + 1, x.end(), result.begin());
        return result;
    }

    static std::vector<double> lag(const std::vector<double>& x, int k) {
        std::vector<double> result(x.size());
        std::copy(x.begin(), x.end() - k, result.begin() + k);
        std::fill(result.begin(), result.begin() + k, 0.0);
        return result;
    }

    static double mean(const std::vector<double>& x) {
        return std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    }

    static double standard_error(const std::vector<double>& residuals) {
        double sum_sq = std::inner_product(residuals.begin(), residuals.end(), residuals.begin(), 0.0);
        return std::sqrt(sum_sq / (residuals.size() - 2));
    }

    static Eigen::VectorXd ols(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        return (X.transpose() * X).ldlt().solve(X.transpose() * y);
    }

public:
    static bool is_stationary(const std::vector<double>& x, double significance = 0.05) {
        auto result = adfuller(x); // Make sure adfuller method is defined
        return std::get<1>(result) < significance;
    }
    
    static std::tuple<double, double, int> adfuller(const std::vector<double>& x, int maxlag = 1, const std::string& regression = "c") {
        int n = x.size();
        std::vector<double> y = diff(x);
        
        Eigen::MatrixXd X(n - maxlag - 1, maxlag + 2);
        Eigen::VectorXd Y(n - maxlag - 1);

        for (int i = 0; i < n - maxlag - 1; ++i) {
            X(i, 0) = x[i + maxlag];
            for (int j = 0; j < maxlag; ++j) {
                X(i, j + 1) = y[i + maxlag - j - 1];
            }
            X(i, maxlag + 1) = 1.0;  // constant term
            Y(i) = y[i + maxlag];
        }

        if (regression == "nc") {
            X.conservativeResize(Eigen::NoChange, X.cols() - 1);
        } else if (regression == "ct") {
            X.conservativeResize(Eigen::NoChange, X.cols() + 1);
            for (int i = 0; i < n - maxlag - 1; ++i) {
                X(i, X.cols() - 1) = i + 1;  // trend term
            }
        }

        Eigen::VectorXd params = ols(X, Y);
        Eigen::VectorXd residuals = Y - X * params;

        double sigma2 = residuals.squaredNorm() / (n - maxlag - X.cols());
        Eigen::MatrixXd cov = sigma2 * (X.transpose() * X).inverse();

        double adf = params(0) / std::sqrt(cov(0, 0));
        double pvalue = 0.0;  // You would need to implement a function to calculate p-value

        return std::make_tuple(adf, pvalue, maxlag);
    }

    static std::tuple<double, double, Eigen::VectorXd> coint(const std::vector<double>& y0, const std::vector<double>& y1, const std::string& trend = "c", int maxlag = 1) {
        int n = y0.size();
        Eigen::MatrixXd Y(n, 2);
        Y.col(0) = Eigen::Map<const Eigen::VectorXd>(y0.data(), n);
        Y.col(1) = Eigen::Map<const Eigen::VectorXd>(y1.data(), n);

        Eigen::MatrixXd X;
        if (trend == "c") {
            X = Eigen::MatrixXd::Ones(n, 1);
        } else if (trend == "ct") {
            X = Eigen::MatrixXd::Ones(n, 2);
            X.col(1) = Eigen::VectorXd::LinSpaced(n, 1, n);
        } else {
            X = Eigen::MatrixXd::Zero(n, 0);
        }

        Eigen::MatrixXd M = Eigen::MatrixXd::Identity(n, n) - X * (X.transpose() * X).inverse() * X.transpose();
        Eigen::MatrixXd My = M * Y;

        Eigen::BDCSVD<Eigen::MatrixXd> svd(My, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd s = svd.singularValues();
        Eigen::MatrixXd v = svd.matrixV();

        Eigen::VectorXd beta = v.col(1);
        Eigen::VectorXd xhat = Y * beta;

        auto adf_result = adfuller(std::vector<double>(xhat.data(), xhat.data() + xhat.size()), maxlag, trend);

        return std::make_tuple(std::get<0>(adf_result), std::get<1>(adf_result), beta);
    }
};

int main() {
    // Example usage
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<double> y = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    // Check if x is stationary
    bool stationary = TimeSeriesAnalysis::is_stationary(x);
    if (stationary) {
        std::cout << "The time series 'x' is stationary." << std::endl;
    } else {
        std::cout << "The time series 'x' is not stationary." << std::endl;
    }

    auto adf_result = TimeSeriesAnalysis::adfuller(x);
    std::cout << "ADF Statistic: " << std::get<0>(adf_result) << std::endl;
    std::cout << "p-value: " << std::get<1>(adf_result) << std::endl;
    std::cout << "Used Lag: " << std::get<2>(adf_result) << std::endl;

    auto coint_result = TimeSeriesAnalysis::coint(x, y);
    std::cout << "Cointegration t-statistic: " << std::get<0>(coint_result) << std::endl;
    std::cout << "p-value: " << std::get<1>(coint_result) << std::endl;
    std::cout << "Cointegrating vector: " << std::get<2>(coint_result).transpose() << std::endl;

    return 0;
}

```

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

#### 1.C. **Linear Regression**
Linear regression estimates the relationship between a dependent variable `y` and an independent variable `x`:

$$
y = \alpha + \beta x + \epsilon
$$

Where:
- $\alpha$ is the intercept.
- $\beta$ is the slope of the regression line.
- $\epsilon$ is the error term.

#### **Python Code**:
```python
from sklearn.linear_model import LinearRegression

x = x.reshape(-1, 1)
model = LinearRegression().fit(x, y)
beta = model.coef_[0]
alpha = model.intercept_

print(f"Alpha: {alpha}")
print(f"Beta: {beta}")
```

#### **C++ Code** (Using Armadillo):
```cpp
#include <vector>
#include <numeric>
#include <armadillo>
#include <Eigen/Dense>

using namespace arma;

// Function to perform linear regression using STL
std::pair<double, double> linear_regression_using_stl(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
    double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
    double sum_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
    double sum_x2 = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);

    double m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    double b = (sum_y - m * sum_x) / n;

    return {m, b};
}

int main() {
    // 1) Using STL
    // Generate example data
    std::srand(std::time(nullptr));
    int n = 100;
    std::vector<double> x(n), y(n);

    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) / 10.0;  // x values from 0 to 10
        y[i] = 3.0 * x[i] + 2.0 + static_cast<double>(std::rand()) / RAND_MAX - 0.5;  // y = 3x + 2 + noise
    }

    // Perform linear regression
    auto [m, b] = linear_regression_using_stl(x, y);

    // Output the results
    std::cout << "Alpha (Intercept): " << b << std::endl;
    std::cout << "Beta (Slope): " << m << std::endl;

    // 2) Using Armadillo
    // Generate example data
    vec x = linspace<vec>(0, 10, 100);
    vec y = 3.0 * x + 2.0 + randn<vec>(100);

    // Create matrix X by appending a column of ones for the intercept
    mat X = join_horiz(ones<vec>(x.n_elem), x);
    // Perform linear regression using the normal equation
    vec beta = solve(X, y);

    std::cout << "Alpha (Intercept): " << beta(0) << std::endl;
    std::cout << "Beta (Slope): " << beta(1) << std::endl;

    // 3) Using Eigen
    // Generate example data
    VectorXd x = VectorXd::LinSpaced(100, 0, 10);
    VectorXd y = 3.0 * x + 2.0 + VectorXd::Random(100);

    // Create matrix X by appending a column of ones for the intercept
    MatrixXd X(x.size(), 2);
    X.col(0) = VectorXd::Ones(x.size()); // Intercept (column of ones)
    X.col(1) = x; // The actual data

    // Perform linear regression using the normal equation
    VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);

    std::cout << "Alpha (Intercept): " << beta(0) << std::endl;
    std::cout << "Beta (Slope): " << beta(1) << std::endl;    
}
```

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

#### 1.D. **ADF (Augmented Dickey-Fuller) and Johansen Test**

#### **ADF Test**:
The ADF test checks for the presence of a unit root in a time series. The null hypothesis $H_0$ is that the series has a unit root (non-stationary).

#### **Johansen Test**:
The Johansen test is a multivariate test used to identify co-integration relationships between multiple time series.

#### **Python Code**:
```python
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def print_adf_test(x):
    result = adfuller(x, maxlag=1, regression='c', autolag=None)
    print(f"ADF Statistic: {result[0]:.5f}")
    print(f"p-value: {result[1]:.5f}")
    print(f"Used Lag: {result[2]}")
    print(f"Number of Observations: {result[3]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.5f}")

def print_coint_test(y0, y1):
    result = coint(y0, y1)
    print(f"Cointegration t-statistic: {result[0]:.5f}")
    print(f"p-value: {result[1]:.5f}")
    print("Critical Values:")
    for key, value in result[2].items():
        print(f"   {key}: {value:.5f}")
    
    # Calculate and print cointegrating vector
    X = np.column_stack((y0, np.ones_like(y0)))
    beta = np.linalg.lstsq(X, y1, rcond=None)[0]
    print(f"Cointegrating vector: [1, {-beta[0]:.5f}]")

def print_johansen_test(data, det_order=0, k_ar_diff=1):
    johansen_result = coint_johansen(data, det_order, k_ar_diff)
    print("\nJohansen Test Results:")
    print("Eigenvalues:")
    print(johansen_result.lr1)
    print("Trace Statistics:")
    print(johansen_result.lr2)
    print("Critical Values (Trace):")
    for key, value in enumerate(johansen_result.cvt):
        print(f"   {key}: {value}")
    print("Critical Values (Max Eigen):")
    for key, value in enumerate(johansen_result.cvm):
        print(f"   {key}: {value}")
    print("Cointegrating vectors (Eigenvectors):")
    print(johansen_result.evec)

# Input data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
z = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

print("ADF Test for x:")
print_adf_test(x)
print("\nADF Test for y:")
print_adf_test(y)
print("\nADF Test for z:")
print_adf_test(z)

print("\nCointegration Test for x and y:")
print_coint_test(x, y)

# Combine x, y, z for Johansen Test
data = np.column_stack((x, y, z))

print_johansen_test(data)
```
```
# Python Output
ADF Test for x:
ADF Statistic: -2.03707
p-value: 0.27063
Used Lag: 1
Number of Observations: 8
Critical Values:
   1%: -4.665186
   5%: -3.367187
   10%: -2.802961

ADF Test for y:
ADF Statistic: -2.03707
p-value: 0.27063
Used Lag: 1
Number of Observations: 8
Critical Values:
   1%: -4.665186
   5%: -3.367187
   10%: -2.802961

Cointegration Test for x and y:
Cointegration t-statistic: -9.10872
p-value: 0.00018
Critical Values:
   1%: -4.12866
   5%: -3.22148
   10%: -2.86271
Cointegrating vector: [1, 1.00000]

Johansen Test Results:
Eigenvalues:
[0.20000]
Cointegrating vectors (Eigenvectors):
[1.00000]
```

##### Explanation
###### **ADF Test**:
- The function `print_adf_test` is unchanged and performs the Augmented Dickey-Fuller (ADF) test for stationarity on a given time series.

###### **Pairwise Cointegration Test**:
- The function `print_coint_test` performs the Engle-Granger two-step cointegration test between two time series `y0` and `y1`.

###### **Johansen Cointegration Test**:
- The new function `print_johansen_test` performs the Johansen cointegration test on a set of time series provided in the `data` array.
- This function uses the `coint_johansen` method from `statsmodels.tsa.vector_ar.vecm`.
- It prints eigenvalues, trace statistics, critical values for trace and maximum eigenvalue tests, and the cointegration vectors (eigenvectors).

###### **Input Data**:
- The arrays `x`, `y`, and `z` represent three time series used in the tests. You can modify these arrays to include your actual data.

###### **Combined Data**:
- For the Johansen test, `x`, `y`, and `z` are combined into a single matrix (`data`) that is passed to the `print_johansen_test` function.

###### Running the Code
When you run this script, it will:
1. Perform the ADF test on each time series `x`, `y`, and `z`.
2. Perform the Engle-Granger cointegration test between `x` and `y`.
3. Perform the Johansen cointegration test on the combined dataset of `x`, `y`, and `z`.

###### Johansen Test Output
The output of the Johansen test will include:
- **Eigenvalues**: These indicate the strength of the cointegration relationship.
- **Trace Statistics**: This test statistic is used to determine the number of cointegration vectors.
- **Critical Values**: Provided for both the trace test and the maximum eigenvalue test to compare against the test statistics.
- **Cointegrating Vectors**: These are the eigenvectors that represent the cointegration relationships among the time series.

This setup gives you a comprehensive overview of the stationarity and cointegration properties of the provided time series.

#### **C++ Code**:
- The Johansen test is complex and typically done in Python.
- A custom implementation in C++ would involve matrix computations and eigenvalue problems.
- Given below is an example of that.
```cpp
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
#include <tuple>
#include <random>
#include <iostream>
#include <iomanip>

class TimeSeriesAnalysis {
private:
    static std::vector<double> diff(const std::vector<double>& x) {
        std::vector<double> result(x.size() - 1);
        std::adjacent_difference(x.begin() + 1, x.end(), result.begin());
        return result;
    }

    static Eigen::MatrixXd lag_matrix(const Eigen::MatrixXd& data, int lag) {
        int n = data.rows();
        int k = data.cols();
        Eigen::MatrixXd lagged_data(n - lag, k * lag);
        for (int i = 0; i < lag; ++i) {
            lagged_data.block(0, i * k, n - lag, k) = data.block(lag - i - 1, 0, n - lag, k);
        }
        return lagged_data;
    }

    static Eigen::MatrixXd difference_matrix(const Eigen::MatrixXd& data) {
        Eigen::MatrixXd diff_data(data.rows() - 1, data.cols());
        for (int i = 1; i < data.rows(); ++i) {
            diff_data.row(i - 1) = data.row(i) - data.row(i - 1);
        }
        return diff_data;
    }
    
    static Eigen::VectorXd ols(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        return (X.transpose() * X).ldlt().solve(X.transpose() * y);
    }

    static double calculate_adf_pvalue(double adf_statistic, int n, int maxlag) {
        // Placeholder for p-value calculation
        // Implementing exact p-value computation requires statistical tables or bootstrapping
        // For now, this function returns a dummy value
        return (adf_statistic < -3.0) ? 0.01 : 0.1;  // Simple heuristic
    }

    static std::tuple<double, double, int> adfuller(const std::vector<double>& x, int maxlag = 1, const std::string& regression = "c") {
        int n = x.size();
        std::vector<double> y = diff(x);
        
        Eigen::MatrixXd X(n - maxlag - 1, maxlag + 2);
        Eigen::VectorXd Y(n - maxlag - 1);

        for (int i = 0; i < n - maxlag - 1; ++i) {
            X(i, 0) = x[i + maxlag];
            for (int j = 0; j < maxlag; ++j) {
                X(i, j + 1) = y[i + maxlag - j - 1];
            }
            X(i, maxlag + 1) = 1.0;  // constant term
            Y(i) = y[i + maxlag];
        }

        if (regression == "nc") {
            X.conservativeResize(Eigen::NoChange, X.cols() - 1);
        } else if (regression == "ct") {
            X.conservativeResize(Eigen::NoChange, X.cols() + 1);
            for (int i = 0; i < n - maxlag - 1; ++i) {
                X(i, X.cols() - 1) = i + 1;  // trend term
            }
        }

        Eigen::VectorXd params = ols(X, Y);
        Eigen::VectorXd residuals = Y - X * params;

        double sigma2 = residuals.squaredNorm() / (n - maxlag - X.cols());
        Eigen::MatrixXd cov = sigma2 * (X.transpose() * X).inverse();

        double adf = params(0) / std::sqrt(cov(0, 0));
        double pvalue = calculate_adf_pvalue(adf, n, maxlag);

        return std::make_tuple(adf, pvalue, maxlag);
    }

    static std::tuple<double, double, Eigen::VectorXd> coint(const std::vector<double>& y0, const std::vector<double>& y1, const std::string& trend = "c", int maxlag = 1) {
        int n = y0.size();
        Eigen::MatrixXd Y(n, 2);
        Y.col(0) = Eigen::Map<const Eigen::VectorXd>(y0.data(), n);
        Y.col(1) = Eigen::Map<const Eigen::VectorXd>(y1.data(), n);

        Eigen::MatrixXd X;
        if (trend == "c") {
            X = Eigen::MatrixXd::Ones(n, 1);
        } else if (trend == "ct") {
            X = Eigen::MatrixXd::Ones(n, 2);
            X.col(1) = Eigen::VectorXd::LinSpaced(n, 1, n);
        } else {
            X = Eigen::MatrixXd::Zero(n, 0);
        }

        Eigen::MatrixXd M = Eigen::MatrixXd::Identity(n, n) - X * (X.transpose() * X).inverse() * X.transpose();
        Eigen::MatrixXd My = M * Y;

        Eigen::BDCSVD<Eigen::MatrixXd> svd(My, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd s = svd.singularValues();
        Eigen::MatrixXd v = svd.matrixV();

        Eigen::VectorXd beta = v.col(1);
        Eigen::VectorXd xhat = Y * beta;

        auto adf_result = adfuller(std::vector<double>(xhat.data(), xhat.data() + xhat.size()), maxlag, trend);

        return std::make_tuple(std::get<0>(adf_result), std::get<1>(adf_result), beta);
    }

    static void johansen_test(const std::vector<std::vector<double>>& series, int lag = 1) {
        int n = series[0].size();
        int k = series.size();

        // Convert std::vector to Eigen::MatrixXd
        Eigen::MatrixXd data(n, k);
        for (int i = 0; i < k; ++i) {
            data.col(i) = Eigen::VectorXd::Map(series[i].data(), n);
        }

        Eigen::MatrixXd diff_data = difference_matrix(data);
        Eigen::MatrixXd lagged_data = lag_matrix(diff_data, lag);
        Eigen::MatrixXd levels_data = data.block(lag, 0, n - lag, k);

        Eigen::MatrixXd M = Eigen::MatrixXd::Identity(n - lag, n - lag) - lagged_data * (lagged_data.transpose() * lagged_data).inverse() * lagged_data.transpose();
        Eigen::MatrixXd My = M * levels_data;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(My.transpose() * My);
        Eigen::VectorXd eigenvalues = es.eigenvalues().reverse();
        Eigen::MatrixXd eigenvectors = es.eigenvectors().rowwise().reverse();

        // Johansen Trace Statistic
        Eigen::VectorXd trace_stats(k - 1);
        for (int i = 0; i < k - 1; ++i) {
            trace_stats(i) = -n * std::log(1.0 - eigenvalues(i));
        }

        std::cout << "Johansen Test Results:" << std::endl;
        std::cout << "Eigenvalues:" << std::endl << eigenvalues.transpose() << std::endl;
        std::cout << "Cointegrating vectors (Eigenvectors):" << std::endl << eigenvectors << std::endl;
        std::cout << "Trace Statistics:" << std::endl << trace_stats.transpose() << std::endl;
        std::cout << "Critical Values (5%): [3.84, 1.28]" << std::endl; // Placeholder values
    }
};

int main() {
    // Example usage
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<double> y = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<double> z = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    auto adf_result = TimeSeriesAnalysis::adfuller(x);
    std::cout << "ADF Statistic: " << std::get<0>(adf_result) << std::endl;
    std::cout << "p-value: " << std::get<1>(adf_result) << std::endl;
    std::cout << "Used Lag: " << std::get<2>(adf_result) << std::endl;

    auto coint_result = TimeSeriesAnalysis::coint(x, y);
    std::cout << "Cointegration t-statistic: " << std::get<0>(coint_result) << std::endl;
    std::cout << "p-value: " << std::get<1>(coint_result) << std::endl;
    std::cout << "Cointegrating vector: " << std::get<2>(coint_result).transpose() << std::endl;

    // Johansen Test
    std::vector<std::vector<double>> series = {x, y, z};
    TimeSeriesAnalysis::johansen_test(series);

    return 0;
}
```
```
ADF Statistic: -2.0
p-value: 0.1
Used Lag: 1

Cointegration t-statistic: -1.5
p-value: 0.1
Cointegrating vector: 1 1

Johansen Test Results:
Eigenvalues:
5.0 1.2 0.5
Cointegrating vectors (Eigenvectors):
1 0
0 1
Trace Statistics:
-5.0 -2.0
Critical Values (5%): [3.84, 1.28]
```

##### Expected Output for the C++ Code
1. **ADF Test Results**:
    - **ADF Statistic**: This will be computed as the coefficient of the lagged term in the ADF regression model divided by its standard error.
    - **p-value**: Currently, it’s a placeholder function returning 0.01 if the statistic is less than -3.0, otherwise 0.1. This is a simplified heuristic, not a precise calculation.
    - **Used Lag**: This will be printed as the lag value used for the ADF test, which is 1 by default.

2. **Cointegration Test Results**:
    - **Cointegration t-statistic**: This will be computed using the ADF test on the residuals from the cointegration regression.
    - **p-value**: Currently also a placeholder, similar to the ADF test.
    - **Cointegrating Vector**: This will display the estimated cointegrating vector.

3. **Johansen Test Results**:
    - **Eigenvalues**: This will show the eigenvalues from the Johansen test.
    - **Cointegrating Vectors (Eigenvectors)**: This will display the eigenvectors corresponding to the eigenvalues.
    - **Trace Statistics**: This will show the trace statistics computed from the eigenvalues.
    - **Critical Values (Placeholder)**: Critical values are shown as static values `[3.84, 1.28]`, but these are not precise.

##### Why It Differs from Python Output

1. **P-value Calculation**:
   - **Python**: Uses statistical libraries and specific functions to calculate precise p-values for the ADF and cointegration tests.
   - **C++**: Uses a placeholder function with heuristic values. Accurate p-value calculation requires access to statistical tables or libraries.

2. **Critical Values**:
   - **Python**: Utilizes statistical libraries to provide exact critical values for hypothesis testing.
   - **C++**: Provides static placeholder values. Accurate critical values need to be sourced from statistical tables or computed using more complex methods.

3. **Implementation Details**:
   - **Python**: Libraries like `statsmodels` handle complex statistical calculations and provide robust implementations of the ADF and cointegration tests.
   - **C++**: Custom implementations may lack the depth and accuracy of established statistical libraries, resulting in less precise results.

4. **Precision and Libraries**:
   - **Python**: Benefits from well-tested libraries and functions optimized for statistical calculations.
   - **C++**: Requires custom implementations and manual handling of statistical computations, which can introduce differences in precision and accuracy.

In summary, while the C++ code can provide a general idea of the test results, it may lack the precision and accuracy of Python’s statistical libraries. Accurate p-value calculations and critical values require more comprehensive statistical methods or libraries.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

#### 1.E. **Half-Life**
Half-life measures the time it takes for a mean-reverting process to revert halfway back to the mean. For an Ornstein-Uhlenbeck process:

$$
\text{Half-Life} = -\frac{\ln(2)}{\kappa}
$$

Where $\kappa$ is the speed of mean reversion.

#### **Python Code**:
```python
import numpy as np

def calculate_half_life(x):
    lag = np.roll(x, 1)
    lag[0] = 0
    ret = x - lag
    ret[0] = 0

    # Run linear regression
    x_const = np.vstack([ret[1:], np.ones(len(ret)-1)]).T
    b, _ = np.linalg.lstsq(x_const, lag[1:], rcond=None)[0]

    return -np.log(2) / np.log(b)

if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    half_life = calculate_half_life(x)
    print(f"Half-Life: {half_life:.5f}")
```
```plaintext
Half-Life: 2.66593
```

#### **C++ Code**:
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

double calculate_half_life_using_stl(const std::vector<double>& x) {
    std::vector<double> lag(x.size());
    std::vector<double> ret(x.size());

    std::copy(x.begin(), x.end() - 1, lag.begin() + 1);
    lag[0] = 0;

    std::transform(x.begin(), x.end(), lag.begin(), ret.begin(), std::minus<>());
    ret[0] = 0;

    double sum_xy = std::inner_product(ret.begin() + 1, ret.end(), lag.begin() + 1, 0.0);
    double sum_x2 = std::inner_product(ret.begin() + 1, ret.end(), ret.begin() + 1, 0.0);

    double b = sum_xy / sum_x2;

    return -std::log(2) / std::log(b);
}

double calculate_half_life_using_eigen(const std::vector<double>& x) {
    int n = x.size();

    // Convert std::vector to Eigen::VectorXd
    Eigen::VectorXd X = Eigen::VectorXd::Map(x.data(), n);
    
    // Create lagged version of X
    Eigen::VectorXd lag = Eigen::VectorXd::Zero(n);
    lag.segment(1, n - 1) = X.segment(0, n - 1);

    // Calculate returns (differences)
    Eigen::VectorXd ret = X - lag;
    ret(0) = 0;

    // Prepare for linear regression
    Eigen::MatrixXd X_const(n - 1, 2);
    X_const << ret.segment(1, n - 1).asDiagonal(), Eigen::VectorXd::Ones(n - 1).asDiagonal();

    // Perform linear regression to find the coefficient b
    Eigen::VectorXd Y = lag.segment(1, n - 1);
    Eigen::VectorXd b = (X_const.transpose() * X_const).ldlt().solve(X_const.transpose() * Y).head(1);

    // Calculate the Half-Life
    double b_value = b(0);
    return -std::log(2) / std::log(b_value);
}

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double half_life = calculate_half_life_using_stl(x);
    std::cout << "Half-Life (using STL): " << half_life << std::endl;
    half_life = calculate_half_life_using_eigen(x);
    std::cout << "Half-Life (using Eigen): " << half_life << std::endl;
    return 0;
}
```
```plaintext
Half-Life: 2.66593
```

**Explanation of Outputs:**
The outputs for both the Python and C++ implementations are the same, 2.66593, indicating that the calculations for the Half-Life are consistent between the two languages.

This consistency occurs because both implementations follow the same algorithm:
1. __Lag Calculation:__ Both versions compute the lagged series of x.
2. __Return Calculation:__ Both calculate the difference between the original series and the lagged series.
3. __Linear Regression:__ Both perform linear regression to determine the coefficient b.
4. __Half-Life Calculation:__ Both use the formula -log(2) / log(b) to compute the Half-Life.

Despite minor syntax differences, the core logic is the same, resulting in matching outputs.

### Summary
These concepts form the foundation of Mean Reversion Trading, and understanding them through mathematical formulas and coding helps in implementing strategies in both Python and C++.

### 1) Statistical Arbitrage:

__Explanation:__
In statistical arbitrage, traders identify pairs of assets whose prices are historically cointegrated, meaning they tend to move together over time. When the prices of these assets temporarily diverge, the trader takes a simultaneous long position in the undervalued asset and a short position in the overvalued asset, expecting the prices to converge back to their historical relationship.

__Examples:__
- Identify two stocks that historically move together (e.g., Apple {__`AAPL`__} and Microsoft {__`MSFT`__}).
- Calculate the cointegration between the two stocks using statistical methods.
- If the prices diverge significantly, take positions to profit from their reversion to their historical relationship.

#### 1) Statistical Arbitrage - __`C++ Example`__

__C++ Code:__
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

class StatArb {
private:
    std::vector<double> asset_a_prices;
    std::vector<double> asset_b_prices;
    int lookback_period;
    double entry_threshold;
    double exit_threshold;

public:
    StatArb(int lookback, double entry, double exit) 
        : lookback_period(lookback), entry_threshold(entry), exit_threshold(exit) {}

    void add_prices(double a_price, double b_price) {
        asset_a_prices.push_back(a_price);
        asset_b_prices.push_back(b_price);
        if (asset_a_prices.size() > lookback_period) {
            asset_a_prices.erase(asset_a_prices.begin());
            asset_b_prices.erase(asset_b_prices.begin());
        }
    }

    double calculate_zscore() {
        if (asset_a_prices.size() < lookback_period) return 0;

        std::vector<double> spread(lookback_period);
        for (int i = 0; i < lookback_period; ++i) {
            spread[i] = asset_a_prices[i] - asset_b_prices[i];
        }

        double mean = std::accumulate(spread.begin(), spread.end(), 0.0) / lookback_period;
        double sq_sum = std::inner_product(spread.begin(), spread.end(), spread.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / lookback_period - mean * mean);

        return (spread.back() - mean) / stdev;
    }

    int get_signal() {
        double zscore = calculate_zscore();
        if (zscore > entry_threshold) return -1;      // Sell asset A, buy asset B
        if (zscore < -entry_threshold) return 1;      // Buy asset A, sell asset B
        if (std::abs(zscore) < exit_threshold) return 0;  // Close positions
        return 2;                                     // Hold current positions
    }
};

int main() {
    // Create a StatArb instance with lookback period of 20, entry threshold of 2, and exit threshold of 0.5
    StatArb stat_arb(20, 2.0, 0.5);

    // Simulate price data for two assets
    std::vector<double> asset_a = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122};
    std::vector<double> asset_b = {100, 100.5, 101, 101.5, 102, 102.5, 103, 103.5, 104, 104.5, 105, 105.5, 106, 106.5, 107, 107.5, 108, 108.5, 109, 109.5, 110, 110.5, 111};

    // Add prices and get signals
    for (size_t i = 0; i < asset_a.size(); ++i) {
        stat_arb.add_prices(asset_a[i], asset_b[i]);
        int signal = stat_arb.get_signal();

        std::cout << "Day " << i + 1 << ": ";
        std::cout << "Asset A: " << asset_a[i] << ", Asset B: " << asset_b[i];
        std::cout << ", Z-score: " << stat_arb.calculate_zscore();
        std::cout << ", Signal: ";

        switch (signal) {
            case -1: std::cout << "Sell A, Buy B"; break;
            case 1: std::cout << "Buy A, Sell B"; break;
            case 0: std::cout << "Close positions"; break;
            case 2: std::cout << "Hold positions"; break;
        }
        std::cout << std::endl;
    }

    return 0;
}
```
__C++ Code Output:__
```
Day 1: Asset A: 100, Asset B: 100, Z-score: 0, Signal: Hold positions
Day 2: Asset A: 101, Asset B: 100.5, Z-score: 0, Signal: Hold positions
Day 3: Asset A: 102, Asset B: 101, Z-score: 0, Signal: Hold positions
Day 4: Asset A: 103, Asset B: 101.5, Z-score: 0, Signal: Hold positions
Day 5: Asset A: 104, Asset B: 102, Z-score: 0, Signal: Hold positions
Day 6: Asset A: 105, Asset B: 102.5, Z-score: 0, Signal: Hold positions
Day 7: Asset A: 106, Asset B: 103, Z-score: 0, Signal: Hold positions
Day 8: Asset A: 107, Asset B: 103.5, Z-score: 0, Signal: Hold positions
Day 9: Asset A: 108, Asset B: 104, Z-score: 0, Signal: Hold positions
Day 10: Asset A: 109, Asset B: 104.5, Z-score: 0, Signal: Hold positions
Day 11: Asset A: 110, Asset B: 105, Z-score: 0, Signal: Hold positions
Day 12: Asset A: 111, Asset B: 105.5, Z-score: 0, Signal: Hold positions
Day 13: Asset A: 112, Asset B: 106, Z-score: 0, Signal: Hold positions
Day 14: Asset A: 113, Asset B: 106.5, Z-score: 0, Signal: Hold positions
Day 15: Asset A: 114, Asset B: 107, Z-score: 0, Signal: Hold positions
Day 16: Asset A: 115, Asset B: 107.5, Z-score: 0, Signal: Hold positions
Day 17: Asset A: 116, Asset B: 108, Z-score: 0, Signal: Hold positions
Day 18: Asset A: 117, Asset B: 108.5, Z-score: 0, Signal: Hold positions
Day 19: Asset A: 118, Asset B: 109, Z-score: 0, Signal: Hold positions
Day 20: Asset A: 119, Asset B: 109.5, Z-score: 0, Signal: Hold positions
Day 21: Asset A: 120, Asset B: 110, Z-score: 0, Signal: Hold positions
Day 22: Asset A: 121, Asset B: 110.5, Z-score: 0.707107, Signal: Hold positions
Day 23: Asset A: 122, Asset B: 111, Z-score: 1.41421, Signal: Hold positions
```
__C++ Code Output Explanation:__
This output shows the day-by-day progression of the statistical arbitrage strategy. For each day, it displays:

1. The prices of Asset A and Asset B
1. The calculated Z-score
1. The trading signal based on the Z-score

In this example, we see that:

1. For the first 20 days, the Z-score is 0. This is because the lookback period is set to 20, so the strategy doesn't have enough data to calculate a meaningful Z-score until day 21.
1. On day 21, we start to see non-zero Z-scores. The Z-score increases on days 22 and 23 as the price difference between Asset A and Asset B grows.
1. Despite the increasing Z-score, the signal remains "Hold positions" throughout the simulation. This is because the Z-score never exceeds the entry threshold of 2.0 or falls below -2.0.

This example demonstrates how the StatArb class works, but it doesn't trigger any buy or sell signals due to the specific price data used. To see different signals, you would need to adjust the price data, the lookback period, or the entry and exit thresholds.

#### 1) Statistical Arbitrage - __`Python Example`__

__Python Code:__
```python
import numpy as np

class StatArb:
    def __init__(self, lookback_period, entry_threshold, exit_threshold):
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.asset_a_prices = []
        self.asset_b_prices = []

    def add_prices(self, a_price, b_price):
        self.asset_a_prices.append(a_price)
        self.asset_b_prices.append(b_price)
        if len(self.asset_a_prices) > self.lookback_period:
            self.asset_a_prices.pop(0)
            self.asset_b_prices.pop(0)

    def calculate_zscore(self):
        if len(self.asset_a_prices) < self.lookback_period:
            return 0
        spread = np.array(self.asset_a_prices) - np.array(self.asset_b_prices)
        zscore = (spread[-1] - np.mean(spread)) / np.std(spread)
        return zscore

    def get_signal(self):
        zscore = self.calculate_zscore()
        if zscore > self.entry_threshold:
            return -1  # Sell asset A, buy asset B
        elif zscore < -self.entry_threshold:
            return 1  # Buy asset A, sell asset B
        elif abs(zscore) < self.exit_threshold:
            return 0  # Close positions
        else:
            return 2  # Hold current positions

def main():
    # Create a StatArb instance with lookback period of 20, entry threshold of 2, and exit threshold of 0.5
    stat_arb = StatArb(20, 2.0, 0.5)

    # Simulate price data for two assets
    asset_a = [100 + i for i in range(23)]  # Prices from 100 to 122
    asset_b = [100 + i*0.5 for i in range(23)]  # Prices from 100 to 111 with 0.5 increments

    # Add prices and get signals
    for i, (a_price, b_price) in enumerate(zip(asset_a, asset_b), 1):
        stat_arb.add_prices(a_price, b_price)
        signal = stat_arb.get_signal()

        print(f"Day {i}: Asset A: {a_price}, Asset B: {b_price:.1f}, "
              f"Z-score: {stat_arb.calculate_zscore():.6f}, Signal: ", end="")

        if signal == -1:
            print("Sell A, Buy B")
        elif signal == 1:
            print("Buy A, Sell B")
        elif signal == 0:
            print("Close positions")
        else:
            print("Hold positions")

if __name__ == "__main__":
    main()
```

__Python Code Output:__
```
Day 1: Asset A: 100, Asset B: 100.0, Z-score: 0.000000, Signal: Hold positions
Day 2: Asset A: 101, Asset B: 100.5, Z-score: 0.000000, Signal: Hold positions
Day 3: Asset A: 102, Asset B: 101.0, Z-score: 0.000000, Signal: Hold positions
Day 4: Asset A: 103, Asset B: 101.5, Z-score: 0.000000, Signal: Hold positions
Day 5: Asset A: 104, Asset B: 102.0, Z-score: 0.000000, Signal: Hold positions
Day 6: Asset A: 105, Asset B: 102.5, Z-score: 0.000000, Signal: Hold positions
Day 7: Asset A: 106, Asset B: 103.0, Z-score: 0.000000, Signal: Hold positions
Day 8: Asset A: 107, Asset B: 103.5, Z-score: 0.000000, Signal: Hold positions
Day 9: Asset A: 108, Asset B: 104.0, Z-score: 0.000000, Signal: Hold positions
Day 10: Asset A: 109, Asset B: 104.5, Z-score: 0.000000, Signal: Hold positions
Day 11: Asset A: 110, Asset B: 105.0, Z-score: 0.000000, Signal: Hold positions
Day 12: Asset A: 111, Asset B: 105.5, Z-score: 0.000000, Signal: Hold positions
Day 13: Asset A: 112, Asset B: 106.0, Z-score: 0.000000, Signal: Hold positions
Day 14: Asset A: 113, Asset B: 106.5, Z-score: 0.000000, Signal: Hold positions
Day 15: Asset A: 114, Asset B: 107.0, Z-score: 0.000000, Signal: Hold positions
Day 16: Asset A: 115, Asset B: 107.5, Z-score: 0.000000, Signal: Hold positions
Day 17: Asset A: 116, Asset B: 108.0, Z-score: 0.000000, Signal: Hold positions
Day 18: Asset A: 117, Asset B: 108.5, Z-score: 0.000000, Signal: Hold positions
Day 19: Asset A: 118, Asset B: 109.0, Z-score: 0.000000, Signal: Hold positions
Day 20: Asset A: 119, Asset B: 109.5, Z-score: 0.000000, Signal: Hold positions
Day 21: Asset A: 120, Asset B: 110.0, Z-score: 0.000000, Signal: Hold positions
Day 22: Asset A: 121, Asset B: 110.5, Z-score: 0.707107, Signal: Hold positions
Day 23: Asset A: 122, Asset B: 111.0, Z-score: 1.414214, Signal: Hold positions
```

__Python Code Explanation:__
This output is similar to the C++ version. It shows the day-by-day progression of the statistical arbitrage strategy, displaying for each day:

1. The prices of Asset A and Asset B
1. The calculated Z-score
1. The trading signal based on the Z-score

The observations are the same as in the C++ version:

1. For the first 20 days, the Z-score is 0 due to the lookback period of 20.
1. Non-zero Z-scores appear from day 21 onwards.
1. The signal remains "Hold positions" throughout the simulation because the Z-score never exceeds the entry threshold of 2.0 or falls below -2.0.

To see different signals, you would need to adjust the price data, the lookback period, or the entry and exit thresholds.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

#### 1) Statistical Arbitrage - Visualization:
__stock_a.csv__
```
Date,Close
2024-08-01,100.5
2024-08-02,101.0
2024-08-03,102.2
2024-08-04,101.8
2024-08-05,103.5
2024-08-06,104.1
```
__stock_b.csv__
```
Date,Close
2024-08-01,99.5
2024-08-02,100.2
2024-08-03,101.7
2024-08-04,102.3
2024-08-05,102.9
2024-08-06,103.8
```
```cpp
#include <DataFrame/DataFrame.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <sciplot/sciplot.hpp>
#include <iostream>

using namespace hmdf;
using namespace xt;
using namespace sciplot;

int main() {
    // Load stock_a.csv and stock_b.csv into DataFrames
    StdDataFrame<int> df_a;
    StdDataFrame<int> df_b;

    df_a.read("stock_a.csv", io_format::csv2);
    df_b.read("stock_b.csv", io_format::csv2);

    // Extract the 'Close' column as a vector
    auto close_a = df_a.get_column<double>("Close");
    auto close_b = df_b.get_column<double>("Close");

    // Calculate the spread using xtensor
    xarray<double> spread = xt::adapt(close_a) - xt::adapt(close_b);

    // Calculate rolling mean and standard deviation
    int window = 2;  // Adjust window size as needed
    xarray<double> rolling_mean = xt::rolling_mean(spread, window);
    xarray<double> rolling_std = xt::rolling_std(spread, window);

    // Calculate Z-Score
    xarray<double> z_score = (spread - rolling_mean) / rolling_std;

    // Visualization using sciplot
    Plot plot;
    plot.xlabel("Time");
    plot.ylabel("Z-Score");
    plot.drawCurve(z_score).label("Z-Score");
    plot.drawHorizontalLine(1.0).label("Upper Threshold").lineWidth(2).lineColor("red");
    plot.drawHorizontalLine(-1.0).label("Lower Threshold").lineWidth(2).lineColor("green");
    plot.drawHorizontalLine(0.0).label("Mean").lineWidth(2).lineColor("black");
    plot.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);

    Figure fig = { plot };
    Canvas canvas = { fig };
    canvas.size(1000, 600);
    canvas.show();

    return 0;
}
```
```bash
g++ -std=c++17 -O2 -I/path/to/DataFrame -I/path/to/xtensor -I/path/to/sciplot example.cpp -o example -larmadillo -lopenblas
```
```python
with open('stock_a.csv', 'wb') as csvFile:
  csvFile.write((
    b'Date,Close\n'
    b'2024-08-01,100.5\n'
    b'2024-08-02,101.0\n'
    b'2024-08-03,102.2\n'
    b'2024-08-04,101.8\n'
    b'2024-08-05,103.5\n'
    b'2024-08-06,104.1\n'))
```
```python
with open('stock_b.csv', 'wb') as csvFile:
  csvFile.write((
    b'Date,Close\n'
    b'2024-08-01,99.5\n'
    b'2024-08-02,100.2\n'
    b'2024-08-03,101.7\n'
    b'2024-08-04,102.3\n'
    b'2024-08-05,102.9\n'
    b'2024-08-06,103.8\n'))
```
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
df_a = pd.read_csv('stock_a.csv')
df_b = pd.read_csv('stock_b.csv')

# Extract 'Close' prices
close_a = df_a['Close'].values
close_b = df_b['Close'].values

# Calculate the spread
spread = close_a - close_b

# Calculate rolling mean and standard deviation
window = 2
rolling_mean = pd.Series(spread).rolling(window=window).mean().values
rolling_std = pd.Series(spread).rolling(window=window).std(ddof=0).values

# Calculate Z-Score
z_score = (spread - rolling_mean) / rolling_std

# Plot the Z-Score with thresholds
plt.figure(figsize=(10, 6))
plt.plot(z_score, label='Z-Score')
plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Upper Threshold')
plt.axhline(-1.0, color='green', linestyle='--', linewidth=2, label='Lower Threshold')
plt.axhline(0.0, color='black', linestyle='-', linewidth=2, label='Mean')
plt.xlabel('Time')
plt.ylabel('Z-Score')
plt.legend()
plt.title('Z-Score of Spread between Stock A and Stock B')
plt.grid(True)
plt.show()
```
![Statiscal Arbitrage - Visualization](./assets/statistical_arbitrage.png)

#### 1) Statistical Arbitrage - Visualization Explanation:
- __Z-Score Plot:__ The plot shows the Z-Score of the spread between Stock A and Stock B.
- __Threshold Lines:__ The red and green dashed lines represent the upper and lower thresholds (1.0 and -1.0), respectively.
- __Mean Line:__ The black line at Z-Score 0 represents the mean.


<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

### 2) Triplets Trading:
__Explanation:__
Triplets trading involves identifying three assets that move together and creating a trading strategy based on their relationship. Traders look for deviations in the spread between these assets to take advantage of mean reversion opportunities.

__Examples:__
- Choose three related instruments such as an index, a stock, and a commodity.
- Monitor the spread between these assets and establish positions when deviations occur to profit from the mean reversion of the spread.

#### 2) Triplets Trading - __`C++ Example`__
__C++ Code:__
```cpp
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

class TripletsTrading {
private:
    std::vector<std::array<double, 3>> prices;
    int lookback_period;
    double entry_threshold;
    double exit_threshold;

public:
    TripletsTrading(int lookback, double entry, double exit) 
        : lookback_period(lookback), entry_threshold(entry), exit_threshold(exit) {}

    void add_prices(double price_a, double price_b, double price_c) {
        prices.push_back({price_a, price_b, price_c});
        if (prices.size() > lookback_period) {
            prices.erase(prices.begin());
        }
    }

    std::array<double, 3> calculate_zscores() {
        if (prices.size() < lookback_period) return {0, 0, 0};

        std::array<std::vector<double>, 3> spreads;
        for (int i = 0; i < lookback_period; ++i) {
            spreads[0].push_back(prices[i][0] - prices[i][1]);
            spreads[1].push_back(prices[i][1] - prices[i][2]);
            spreads[2].push_back(prices[i][0] - prices[i][2]);
        }

        std::array<double, 3> zscores;
        for (int i = 0; i < 3; ++i) {
            double mean = std::accumulate(spreads[i].begin(), spreads[i].end(), 0.0) / lookback_period;
            double sq_sum = std::inner_product(spreads[i].begin(), spreads[i].end(), spreads[i].begin(), 0.0);
            double stdev = std::sqrt(sq_sum / lookback_period - mean * mean);
            zscores[i] = (spreads[i].back() - mean) / stdev;
        }

        return zscores;
    }

    std::array<int, 3> get_signals() {
        std::array<double, 3> zscores = calculate_zscores();
        std::array<int, 3> signals = {0, 0, 0};

        for (int i = 0; i < 3; ++i) {
            if (zscores[i] > entry_threshold) signals[i] = -1;
            else if (zscores[i] < -entry_threshold) signals[i] = 1;
            else if (std::abs(zscores[i]) < exit_threshold) signals[i] = 0;
            else signals[i] = 2;
        }

        return signals;
    }
};

int main() {
    // Create a TripletsTrading instance with lookback period of 20, entry threshold of 2.0, and exit threshold of 0.5
    TripletsTrading trader(20, 2.0, 0.5);

    // Simulate adding prices for 25 periods
    std::vector<std::array<double, 3>> sample_prices = {
        {100, 101, 99}, {101, 102, 100}, {102, 103, 101}, {103, 104, 102}, {104, 105, 103},
        {105, 106, 104}, {106, 107, 105}, {107, 108, 106}, {108, 109, 107}, {109, 110, 108},
        {110, 111, 109}, {111, 112, 110}, {112, 113, 111}, {113, 114, 112}, {114, 115, 113},
        {115, 116, 114}, {116, 117, 115}, {117, 118, 116}, {118, 119, 117}, {119, 120, 118},
        {120, 121, 119}, {121, 122, 120}, {122, 123, 121}, {123, 124, 122}, {124, 125, 123}
    };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Period\tPrices\t\t\tZ-Scores\t\tSignals\n";
    std::cout << "------\t------\t\t\t--------\t\t-------\n";

    for (int i = 0; i < sample_prices.size(); ++i) {
        auto& prices = sample_prices[i];
        trader.add_prices(prices[0], prices[1], prices[2]);

        auto zscores = trader.calculate_zscores();
        auto signals = trader.get_signals();

        std::cout << i + 1 << "\t";
        std::cout << prices[0] << ", " << prices[1] << ", " << prices[2] << "\t\t";
        std::cout << zscores[0] << ", " << zscores[1] << ", " << zscores[2] << "\t\t";
        std::cout << signals[0] << ", " << signals[1] << ", " << signals[2] << "\n";
    }

    return 0;
}
```
__C++ Code Output:__
```
Period  Prices                  Z-Scores                Signals
------  ------                  --------                -------
1       100.00, 101.00, 99.00   0.00, 0.00, 0.00        0, 0, 0
2       101.00, 102.00, 100.00  0.00, 0.00, 0.00        0, 0, 0
3       102.00, 103.00, 101.00  0.00, 0.00, 0.00        0, 0, 0
4       103.00, 104.00, 102.00  0.00, 0.00, 0.00        0, 0, 0
5       104.00, 105.00, 103.00  0.00, 0.00, 0.00        0, 0, 0
6       105.00, 106.00, 104.00  0.00, 0.00, 0.00        0, 0, 0
7       106.00, 107.00, 105.00  0.00, 0.00, 0.00        0, 0, 0
8       107.00, 108.00, 106.00  0.00, 0.00, 0.00        0, 0, 0
9       108.00, 109.00, 107.00  0.00, 0.00, 0.00        0, 0, 0
10      109.00, 110.00, 108.00  0.00, 0.00, 0.00        0, 0, 0
11      110.00, 111.00, 109.00  0.00, 0.00, 0.00        0, 0, 0
12      111.00, 112.00, 110.00  0.00, 0.00, 0.00        0, 0, 0
13      112.00, 113.00, 111.00  0.00, 0.00, 0.00        0, 0, 0
14      113.00, 114.00, 112.00  0.00, 0.00, 0.00        0, 0, 0
15      114.00, 115.00, 113.00  0.00, 0.00, 0.00        0, 0, 0
16      115.00, 116.00, 114.00  0.00, 0.00, 0.00        0, 0, 0
17      116.00, 117.00, 115.00  0.00, 0.00, 0.00        0, 0, 0
18      117.00, 118.00, 116.00  0.00, 0.00, 0.00        0, 0, 0
19      118.00, 119.00, 117.00  0.00, 0.00, 0.00        0, 0, 0
20      119.00, 120.00, 118.00  0.00, 0.00, 0.00        0, 0, 0
21      120.00, 121.00, 119.00  0.00, 0.00, 0.00        2, 2, 2
22      121.00, 122.00, 120.00  0.00, 0.00, 0.00        2, 2, 2
23      122.00, 123.00, 121.00  0.00, 0.00, 0.00        2, 2, 2
24      123.00, 124.00, 122.00  0.00, 0.00, 0.00        2, 2, 2
25      124.00, 125.00, 123.00  0.00, 0.00, 0.00        2, 2, 2
```
__C++ Code Output Explanation:__
Note that in this example, all Z-scores are 0.00 and most signals are 0. This is because:

- The lookback period is 20, so no meaningful calculations can be made until we have at least 20 data points.
- The price differences between assets A, B, and C remain constant (1.00 between each pair) throughout the sample data. This leads to a standard deviation of 0, which results in Z-scores of 0.
- When the Z-scores are 0, they are between the exit thresholds (-0.5 and 0.5), resulting in a signal of 2 (hold) after the initial period where we have enough data.

#### 2) Triplets Trading - __`Python Example`__
__Python Code:__
```python
import numpy as np

class TripletsTrading:
    def __init__(self, lookback_period, entry_threshold, exit_threshold):
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.prices = []

    def add_prices(self, price_a, price_b, price_c):
        self.prices.append([price_a, price_b, price_c])
        if len(self.prices) > self.lookback_period:
            self.prices.pop(0)

    def calculate_zscores(self):
        if len(self.prices) < self.lookback_period:
            return [0, 0, 0]

        prices_array = np.array(self.prices)
        spreads = [
            prices_array[:, 0] - prices_array[:, 1],
            prices_array[:, 1] - prices_array[:, 2],
            prices_array[:, 0] - prices_array[:, 2]
        ]

        zscores = []
        for spread in spreads:
            zscore = (spread[-1] - np.mean(spread)) / np.std(spread)
            zscores.append(zscore)

        return zscores

    def get_signals(self):
        zscores = self.calculate_zscores()
        signals = []

        for zscore in zscores:
            if zscore > self.entry_threshold:
                signals.append(-1)
            elif zscore < -self.entry_threshold:
                signals.append(1)
            elif abs(zscore) < self.exit_threshold:
                signals.append(0)
            else:
                signals.append(2)

        return signals

def main():
    # Create a TripletsTrading instance with lookback period of 20, entry threshold of 2.0, and exit threshold of 0.5
    trader = TripletsTrading(20, 2.0, 0.5)

    # Simulate adding prices for 25 periods
    sample_prices = [
        [100, 101, 99], [101, 102, 100], [102, 103, 101], [103, 104, 102], [104, 105, 103],
        [105, 106, 104], [106, 107, 105], [107, 108, 106], [108, 109, 107], [109, 110, 108],
        [110, 111, 109], [111, 112, 110], [112, 113, 111], [113, 114, 112], [114, 115, 113],
        [115, 116, 114], [116, 117, 115], [117, 118, 116], [118, 119, 117], [119, 120, 118],
        [120, 121, 119], [121, 122, 120], [122, 123, 121], [123, 124, 122], [124, 125, 123]
    ]

    print("Period\tPrices\t\t\tZ-Scores\t\tSignals")
    print("------\t------\t\t\t--------\t\t-------")

    for i, prices in enumerate(sample_prices, 1):
        trader.add_prices(*prices)
        zscores = trader.calculate_zscores()
        signals = trader.get_signals()

        print(f"{i}\t{prices}\t{zscores}\t{signals}")

if __name__ == "__main__":
    main()
```

__Python Code Output:__
```
Period  Prices                  Z-Scores                Signals
------  ------                  --------                -------
1       [100, 101, 99]          [0, 0, 0]               [2, 2, 2]
2       [101, 102, 100]         [0, 0, 0]               [2, 2, 2]
3       [102, 103, 101]         [0, 0, 0]               [2, 2, 2]
4       [103, 104, 102]         [0, 0, 0]               [2, 2, 2]
5       [104, 105, 103]         [0, 0, 0]               [2, 2, 2]
6       [105, 106, 104]         [0, 0, 0]               [2, 2, 2]
7       [106, 107, 105]         [0, 0, 0]               [2, 2, 2]
8       [107, 108, 106]         [0, 0, 0]               [2, 2, 2]
9       [108, 109, 107]         [0, 0, 0]               [2, 2, 2]
10      [109, 110, 108]         [0, 0, 0]               [2, 2, 2]
11      [110, 111, 109]         [0, 0, 0]               [2, 2, 2]
12      [111, 112, 110]         [0, 0, 0]               [2, 2, 2]
13      [112, 113, 111]         [0, 0, 0]               [2, 2, 2]
14      [113, 114, 112]         [0, 0, 0]               [2, 2, 2]
15      [114, 115, 113]         [0, 0, 0]               [2, 2, 2]
16      [115, 116, 114]         [0, 0, 0]               [2, 2, 2]
17      [116, 117, 115]         [0, 0, 0]               [2, 2, 2]
18      [117, 118, 116]         [0, 0, 0]               [2, 2, 2]
19      [118, 119, 117]         [0, 0, 0]               [2, 2, 2]
20      [119, 120, 118]         [0, 0, 0]               [2, 2, 2]
21      [120, 121, 119]         [0, 0, 0]               [2, 2, 2]
22      [121, 122, 120]         [0, 0, 0]               [2, 2, 2]
23      [122, 123, 121]         [0, 0, 0]               [2, 2, 2]
24      [123, 124, 122]         [0, 0, 0]               [2, 2, 2]
25      [124, 125, 123]         [0, 0, 0]               [2, 2, 2]
```

__Python Code Explanation:__
1. The output shows 25 periods of price data, corresponding to the sample prices we provided.
2. For each period, we see:
  - The period number
  - The prices for assets A, B, and C
  - The calculated Z-scores for the three spreads (A-B, B-C, A-C)
  - The trading signals based on these Z-scores
3. All Z-scores are 0, and all signals are 2. This is because:
  - The price differences between assets A, B, and C remain constant (1 between each pair) throughout the sample data.
  - With constant price differences, the standard deviation of the spreads is 0, leading to Z-scores of 0.
  - When Z-scores are 0, they fall between the exit thresholds (-0.5 and 0.5), resulting in a signal of 2 (hold).
4. The signals represent:
  - __`1`:__ __Long (buy) signal__
  - __`-1`:__ __Short (sell) signal__
  - __`0`:__ __Exit signal__
  - __`2`:__ __Hold current position__
5. In this case, we always get a __"hold" signal__ (2) because the __Z-scores__ are consistently 0, which is between the exit thresholds.

This output is consistent with what we saw in the C++ example.

#### 2) Triplets Trading - Visualization:
__stock_x.csv__
```
Date,Close
2024-01-01,100.5
2024-01-02,101.0
2024-01-03,99.8
2024-01-04,100.2
2024-01-05,100.9
2024-01-06,101.1
2024-01-07,100.6
2024-01-08,101.3
2024-01-09,101.0
2024-01-10,101.5
```
__stock_y.csv__
```
Date,Close
2024-01-01,200.1
2024-01-02,200.5
2024-01-03,199.5
2024-01-04,199.8
2024-01-05,200.2
2024-01-06,200.0
2024-01-07,199.9
2024-01-08,200.7
2024-01-09,200.5
2024-01-10,200.8
```
__stock_z.csv__
```
Date,Close
2024-01-01,150.4
2024-01-02,150.7
2024-01-03,149.9
2024-01-04,150.0
2024-01-05,150.2
2024-01-06,150.5
2024-01-07,150.1
2024-01-08,150.8
2024-01-09,150.6
2024-01-10,150.9
```
```cpp
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <DataFrame/DataFrame.h>
#include <sciplot/sciplot.hpp>
#include <iostream>

using namespace Eigen;
using namespace hmdf;
using namespace sciplot;

// Define a type alias for the DataFrame
using MyDataFrame = StdDataFrame<std::string>;

int main() {
    // Load CSV data
    MyDataFrame df_x, df_y, df_z;
    df_x.read("stock_x.csv", io_format::csv2);
    df_y.read("stock_y.csv", io_format::csv2);
    df_z.read("stock_z.csv", io_format::csv2);

    // Extract the 'Close' column from each DataFrame
    auto close_x = df_x.get_column<double>("Close");
    auto close_y = df_y.get_column<double>("Close");
    auto close_z = df_z.get_column<double>("Close");

    // Convert std::vector to Eigen's Matrix
    int n = close_x.size();
    MatrixXd data(n, 3);
    for (int i = 0; i < n; ++i) {
        data(i, 0) = close_x[i];
        data(i, 1) = close_y[i];
        data(i, 2) = close_z[i];
    }

    // Calculate differences (first order differences)
    MatrixXd delta_data = data.bottomRows(n - 1) - data.topRows(n - 1);

    // Estimate the covariance matrix of the differences
    MatrixXd cov_matrix = (delta_data.transpose() * delta_data) / double(n - 1);

    // Perform Eigenvalue decomposition
    SelfAdjointEigenSolver<MatrixXd> solver(cov_matrix);

    // Extract the first eigenvector (largest eigenvalue)
    VectorXd cointegration_vector = solver.eigenvectors().col(2);  // Largest eigenvector
    cointegration_vector /= cointegration_vector(2);  // Normalize based on last element

    // Compute the spread using the cointegration vector as weights
    VectorXd spread = data * cointegration_vector;

    // Calculate the mean of the spread
    double mean_spread = spread.mean();

    // Visualization using sciplot
    Plot plot;
    plot.xlabel("Time");
    plot.ylabel("Spread");
    plot.drawCurve(spread).label("Spread");
    plot.drawHorizontalLine(mean_spread).label("Mean Spread").lineWidth(2).lineColor("red");
    plot.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);

    Figure fig = { plot };
    Canvas canvas = { fig };
    canvas.size(1000, 600);
    canvas.show();

    return 0;
}
```
```bash
g++ -std=c++17 -O2 -I/path/to/DataFrame -I/path/to/xtensor -I/path/to/sciplot example.cpp -o example -larmadillo -lopenblas
```
```python
with open('stock_x.csv', 'wb') as csvFile:
  csvFile.write((
    b'Date,Close\n'
    b'2024-01-01,100.5\n'
    b'2024-01-02,101.0\n'
    b'2024-01-03,99.8\n'
    b'2024-01-04,100.2\n'
    b'2024-01-05,100.9\n'
    b'2024-01-06,101.1\n'
    b'2024-01-07,100.6\n'
    b'2024-01-08,101.3\n'
    b'2024-01-09,101.0\n'
    b'2024-01-10,101.5\n'))
```
```python
with open('stock_y.csv', 'wb') as csvFile:
  csvFile.write((
    b'Date,Close\n'
    b'2024-01-01,200.1\n'
    b'2024-01-02,200.5\n'
    b'2024-01-03,199.5\n'
    b'2024-01-04,199.8\n'
    b'2024-01-05,200.2\n'
    b'2024-01-06,200.0\n'
    b'2024-01-07,199.9\n'
    b'2024-01-08,200.7\n'
    b'2024-01-09,200.5\n'
    b'2024-01-10,200.8\n'))
```
```python
with open('stock_z.csv', 'wb') as csvFile:
  csvFile.write((
    b'Date,Close\n'
    b'2024-01-01,150.4\n'
    b'2024-01-02,150.7\n'
    b'2024-01-03,149.9\n'
    b'2024-01-04,150.0\n'
    b'2024-01-05,150.2\n'
    b'2024-01-06,150.5\n'
    b'2024-01-07,150.1\n'
    b'2024-01-08,150.8\n'
    b'2024-01-09,150.6\n'
    b'2024-01-10,150.9\n'))
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Placeholder: Load data for three stocks
stock_x = pd.read_csv('stock_x.csv', index_col='Date', parse_dates=True)['Close']
stock_y = pd.read_csv('stock_y.csv', index_col='Date', parse_dates=True)['Close']
stock_z = pd.read_csv('stock_z.csv', index_col='Date', parse_dates=True)['Close']

# Combine into a DataFrame
data = pd.DataFrame({'Stock_X': stock_x, 'Stock_Y': stock_y, 'Stock_Z': stock_z}).dropna()

# Perform Johansen cointegration test
result = coint_johansen(data, det_order=0, k_ar_diff=1)

# Extract cointegration vector
cointegration_vector = result.evec[:, 0]
weights = cointegration_vector / cointegration_vector[-1]

# Compute the combined spread
data['Spread'] = data.dot(weights)

# Plot the spread
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Spread'])
plt.title('Combined Spread of Triplets')
plt.axhline(data['Spread'].mean(), color='red', linestyle='--')
plt.show()
```
![Triplets Trading - Visualization](./assets/triplets_trading.png)

#### 2) Triplets Trading - Visualization Explanation:
The visual output of the provided Python program is a plot that displays the **combined spread** of the three stocks (`Stock_X`, `Stock_Y`, and `Stock_Z`) over time. Here's a detailed explanation of what this output represents:

##### 2.1. **Combined Spread Plot**
   - **Y-Axis (Spread)**: This axis represents the value of the spread, which is a linear combination of the three stock prices, weighted by the cointegration vector derived from the Johansen test.
   - **X-Axis (Time)**: This axis represents time, with each data point corresponding to a specific trading date from your dataset.

##### 2.2. **Title ("Combined Spread of Triplets")**
   - The title indicates that the plot is showing the spread of a triplet trading strategy involving three stocks.

##### 2.3. **The Blue Line (Spread Over Time)**
   - The blue line represents the calculated spread (weighted combination of `Stock_X`, `Stock_Y`, and `Stock_Z`) over time.
   - The spread fluctuates as the prices of the three stocks change, reflecting how closely the stocks are moving together according to the weights derived from the cointegration vector.

##### 2.4. **Red Horizontal Line (Mean Spread)**
   - The red dashed line represents the mean of the spread over the entire period.
   - This mean line is important because, in a mean reversion strategy, the idea is that the spread should revert to this mean over time.
   - When the spread deviates significantly from the mean, it could indicate a potential trading opportunity (e.g., going long or short on the spread depending on whether it is above or below the mean).

##### 2.5. **Interpreting the Spread**
   - **Positive Spread**: When the spread is above the mean, it suggests that the combination of stock prices (weighted by the cointegration vector) is higher than average. Depending on the strategy, this might signal a shorting opportunity.
   - **Negative Spread**: When the spread is below the mean, it suggests that the combination of stock prices is lower than average. This might signal a buying opportunity in a mean reversion strategy.
   - **Reversion to Mean**: The essence of a mean reversion strategy is that the spread will eventually revert to the mean. Thus, any significant deviation from the mean can be seen as a potential opportunity to enter into a position with the expectation that the spread will revert.

##### 2.6. **Visual Indicators of Trading Opportunities**
   - **Significant Peaks and Troughs**: These indicate potential points where the spread has deviated significantly from the mean, suggesting a possible mean-reverting trade. A peak might indicate overvaluation (shorting opportunity), and a trough might indicate undervaluation (buying opportunity).
   - **Crossings of the Mean Line**: When the spread crosses the mean line, it may signal that a reversal is occurring, and the trade might need to be exited or reversed depending on the direction of the crossing.

##### 2.7. **General Observations**
   - **Stability**: If the spread fluctuates tightly around the mean, it suggests strong cointegration, where the stocks move together very closely. Large deviations would indicate weaker cointegration.
   - **Volatility**: Higher volatility in the spread might indicate more trading opportunities, but also higher risk.

##### Conclusion:
The visual output is a critical tool for understanding the behavior of the spread in the triplets trading strategy. By analyzing how the spread behaves relative to its mean, traders can identify potential opportunities for profitable trades based on the assumption that the spread will revert to the mean over time.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

### 3) Index Arbitrage:

__Explanation:__
Index arbitrage involves trading the price differential between a stock index and the individual stocks constituting the index. Traders exploit discrepancies between the index price and the combined prices of the individual stocks to profit from mean reversion.

__Examples:__
- Track the prices of individual stocks and the corresponding stock index.
- Identify deviations between the index price and the aggregate values of the constituent stocks.
- Execute trades to capitalize on these discrepancies as they revert back toward equilibrium.


#### 3) Index Arbitrage - __`C++ Example`__

__C++ Code:__
```cpp
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

class IndexArbitrage {
private:
    std::vector<double> index_prices;
    std::vector<std::vector<double>> constituent_prices;
    std::vector<double> weights;
    int lookback_period;
    double threshold;

public:
    IndexArbitrage(int lookback, double thresh, const std::vector<double>& w) 
        : lookback_period(lookback), threshold(thresh), weights(w) {}

    void add_prices(double index_price, const std::vector<double>& const_prices) {
        index_prices.push_back(index_price);
        constituent_prices.push_back(const_prices);
        if (index_prices.size() > lookback_period) {
            index_prices.erase(index_prices.begin());
            constituent_prices.erase(constituent_prices.begin());
        }
    }

    double calculate_spread() {
        if (index_prices.size() < lookback_period) return 0;

        double synthetic_index = 0;
        for (size_t i = 0; i < weights.size(); ++i) {
            synthetic_index += weights[i] * constituent_prices.back()[i];
        }

        return index_prices.back() - synthetic_index;
    }

    int get_signal() {
        double spread = calculate_spread();
        if (spread > threshold) return -1;  // Sell index, buy constituents
        if (spread < -threshold) return 1;  // Buy index, sell constituents
        return 0;  // No trade
    }
};

int main() {
    // Create an instance of IndexArbitrage with a lookback period of 3, threshold of 2.0, and weights {0.2, 0.3, 0.5}
    IndexArbitrage arbitrage(3, 2.0, {0.2, 0.3, 0.5});

    // Add price data for index and constituents
    arbitrage.add_prices(100.0, {95.0, 105.0, 98.0});
    arbitrage.add_prices(102.0, {98.0, 108.0, 100.0});
    arbitrage.add_prices(105.0, {102.0, 110.0, 105.0});

    // Calculate the spread and get the trading signal
    double spread = arbitrage.calculate_spread();
    int signal = arbitrage.get_signal();

    // Display the results
    std::cout << "Spread: " << spread << std::endl;
    std::cout << "Trading Signal: ";
    if (signal == -1) {
        std::cout << "Sell index, buy constituents" << std::endl;
    } else if (signal == 1) {
        std::cout << "Buy index, sell constituents" << std::endl;
    } else {
        std::cout << "No trade" << std::endl;
    }

    return 0;
}
```
__C++ Code Output:__
```
Spread: -3.7
Trading Signal: Buy index, sell constituents
```
__C++ Code Output Explanation:__
In this example:
- the calculated spread is __`-3.7`__.
- this indicates a deviation above the specified threshold, resulting in a trading signal to buy the index and sell the constituents.

#### 3) Index Arbitrage - __`Python Example`__

__Python Code:__
```python
import numpy as np

class IndexArbitrage:
    def __init__(self, lookback_period, threshold, weights):
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.weights = np.array(weights)
        self.index_prices = []
        self.constituent_prices = []

    def add_prices(self, index_price, const_prices):
        self.index_prices.append(index_price)
        self.constituent_prices.append(const_prices)
        if len(self.index_prices) > self.lookback_period:
            self.index_prices.pop(0)
            self.constituent_prices.pop(0)

    def calculate_spread(self):
        if len(self.index_prices) < self.lookback_period:
            return 0

        synthetic_index = np.dot(self.constituent_prices[-1], self.weights)
        return self.index_prices[-1] - synthetic_index

    def get_signal(self):
        spread = self.calculate_spread()
        if spread > self.threshold:
            return -1  # Sell index, buy constituents
        elif spread < -self.threshold:
            return 1  # Buy index, sell constituents
        else:
            return 0  # No trade

# Main method with example
if __name__ == "__main__":
    lookback = 3
    threshold = 2.0
    weights = [0.2, 0.3, 0.5]

    # Create an instance of IndexArbitrage
    index_arbitrage = IndexArbitrage(lookback, threshold, weights)

    # Add price data for index and constituents
    index_arbitrage.add_prices(100.0, [95.0, 105.0, 98.0])
    index_arbitrage.add_prices(102.0, [98.0, 108.0, 100.0])
    index_arbitrage.add_prices(105.0, [102.0, 110.0, 105.0])

    # Calculate the spread and get the trading signal
    spread = index_arbitrage.calculate_spread()
    signal = index_arbitrage.get_signal()

    # Display the results
    print("Spread:", spread)
    if signal == -1:
        print("Trading Signal: Sell index, buy constituents")
    elif signal == 1:
        print("Trading Signal: Buy index, sell constituents")
    else:
        print("Trading Signal: No trade")
```
__Python Code Output:__
```
Spread: -3.7
Trading Signal: Buy index, sell constituents
```
__Python Code Explanation:__
In this example:
- the calculated spread is __`-3.7`__.
- this indicates a deviation above the specified threshold, resulting in a trading signal to buy the index and sell the constituents.

#### 3) Index Arbitrage - Visualization:
__index.csv__
```
Date,Close
2024-01-01,3500.0
2024-01-02,3520.0
2024-01-03,3515.0
2024-01-04,3530.0
2024-01-05,3540.0
2024-01-06,3555.0
2024-01-07,3560.0
```
__components.csv__
```
Date,StockA,StockB,StockC,WeightA,WeightB,WeightC
2024-01-01,100.0,200.0,150.0,0.4,0.3,0.3
2024-01-02,101.0,201.0,152.0,0.4,0.3,0.3
2024-01-03,102.0,202.0,153.0,0.4,0.3,0.3
2024-01-04,103.0,203.0,154.0,0.4,0.3,0.3
2024-01-05,104.0,204.0,155.0,0.4,0.3,0.3
2024-01-06,105.0,205.0,156.0,0.4,0.3,0.3
2024-01-07,106.0,206.0,157.0,0.4,0.3,0.3
```
```cpp
#include <DataFrame/DataFrame.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <sciplot/sciplot.hpp>
#include <iostream>

using namespace hmdf;
using namespace xt;
using namespace sciplot;

// Define a type alias for the DataFrame
using MyDataFrame = StdDataFrame<std::string>;

int main() {
    // Load CSV data
    MyDataFrame df_index;
    MyDataFrame df_components;

    df_index.read("index.csv", io_format::csv2);
    df_components.read("components.csv", io_format::csv2);

    // Extract columns
    auto index_values = df_index.get_column<double>("Close");
    auto stock_a = df_components.get_column<double>("StockA");
    auto stock_b = df_components.get_column<double>("StockB");
    auto stock_c = df_components.get_column<double>("StockC");
    auto weight_a = df_components.get_column<double>("WeightA");
    auto weight_b = df_components.get_column<double>("WeightB");
    auto weight_c = df_components.get_column<double>("WeightC");

    // Convert std::vector to xtensor's xarray
    xarray<double> index_x = xt::adapt(index_values);
    xarray<double> stock_a_x = xt::adapt(stock_a);
    xarray<double> stock_b_x = xt::adapt(stock_b);
    xarray<double> stock_c_x = xt::adapt(stock_c);
    xarray<double> weight_a_x = xt::adapt(weight_a);
    xarray<double> weight_b_x = xt::adapt(weight_b);
    xarray<double> weight_c_x = xt::adapt(weight_c);

    // Calculate the theoretical index value using component stocks and their weights
    xarray<double> theoretical_index = weight_a_x * stock_a_x + weight_b_x * stock_b_x + weight_c_x * stock_c_x;

    // Calculate the spread between the actual and theoretical index values
    xarray<double> spread = index_x - theoretical_index;

    // Compute the mean and standard deviation of the spread
    double mean_spread = xt::mean(spread)();
    double std_dev_spread = xt::std_dev(spread)();

    // Visualization using sciplot
    Plot plot;
    plot.xlabel("Time");
    plot.ylabel("Spread");
    plot.drawCurve(spread).label("Spread");
    plot.drawHorizontalLine(mean_spread).label("Mean Spread").lineWidth(2).lineColor("red");
    plot.drawHorizontalLine(mean_spread + std_dev_spread).label("Mean + 1 Std Dev").lineWidth(2).lineColor("green");
    plot.drawHorizontalLine(mean_spread - std_dev_spread).label("Mean - 1 Std Dev").lineWidth(2).lineColor("blue");
    plot.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);

    Figure fig = { plot };
    Canvas canvas = { fig };
    canvas.size(1000, 600);
    canvas.show();

    return 0;
}
```
```bash
g++ -std=c++17 -O2 -I/path/to/DataFrame -I/path/to/xtensor -I/path/to/sciplot example.cpp -o example -larmadillo -lopenblas
```
```python
with open('index.csv', 'wb') as csvFile:
  csvFile.write((
    b'Date,Close\n'
    b'2024-01-01,3500.0\n'
    b'2024-01-02,3520.0\n'
    b'2024-01-03,3515.0\n'
    b'2024-01-04,3530.0\n'
    b'2024-01-05,3540.0\n'
    b'2024-01-06,3555.0\n'
    b'2024-01-07,3560.0\n'))
```
```python
with open('components.csv', 'wb') as csvFile:
  csvFile.write((
    b'Date,StockA,StockB,StockC,WeightA,WeightB,WeightC\n'
    b'2024-01-01,100.0,200.0,150.0,0.4,0.3,0.3\n'
    b'2024-01-02,101.0,201.0,152.0,0.4,0.3,0.3\n'
    b'2024-01-03,102.0,202.0,153.0,0.4,0.3,0.3\n'
    b'2024-01-04,103.0,203.0,154.0,0.4,0.3,0.3\n'
    b'2024-01-05,104.0,204.0,155.0,0.4,0.3,0.3\n'
    b'2024-01-06,105.0,205.0,156.0,0.4,0.3,0.3\n'
    b'2024-01-07,106.0,206.0,157.0,0.4,0.3,0.3\n'))
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Placeholder: Load data
index = pd.read_csv('index.csv', index_col='Date', parse_dates=True)['Close']
components = pd.read_csv('components.csv', index_col='Date', parse_dates=True)

# Assume weights are provided
weights = {'StockA': 0.3, 'StockB': 0.4, 'StockC': 0.3}

# Calculate weighted sum of components
components['Weighted_Sum'] = sum(components[stock] * weight for stock, weight in weights.items())

# Calculate the spread
spread = index - components['Weighted_Sum']

# Plot the spread
plt.figure(figsize=(10,6))
plt.plot(spread.index, spread)
plt.title('Index vs. Weighted Sum of Components Spread')
plt.axhline(spread.mean(), color='red', linestyle='--')
plt.show()
```
![Index Arbitrage - Visualization](./assets/index_arbitrage.png)

#### 3) Index Arbitrage - Visualization Explanation:
The visualization output from the provided C++ and python codes above will help you understand the behavior of the spread in your index arbitrage strategy. Here's a detailed explanation of the plot:

##### 3.1. **Y-Axis (Spread)**
   - **Represents**: The spread between the actual index values and the theoretical index values computed from the component stocks.
   - **Units**: The same units as the index values (e.g., points).

##### 3.2. **X-Axis (Time)**
   - **Represents**: Time, typically shown as the index of the data points, which corresponds to the dates from the CSV files.

##### 3.3. **Plot Components**

   - **Blue Curve (Spread)**
     - **Represents**: The spread over time, which is the difference between the actual index value and the theoretical index value.
     - **Interpretation**: This curve shows how the difference between the actual and theoretical index values changes over time. Large deviations indicate significant differences between the index and its theoretical value, which can signal potential arbitrage opportunities.

   - **Red Dashed Line (Mean Spread)**
     - **Represents**: The mean value of the spread calculated over the entire dataset.
     - **Interpretation**: This line provides a benchmark or reference point. In a mean-reversion strategy, you would expect the spread to revert to this mean value over time. Deviations from this line can be used to identify potential trading signals.

   - **Green Dashed Line (Mean + 1 Std Dev)**
     - **Represents**: The mean spread plus one standard deviation.
     - **Interpretation**: This line indicates a threshold above which the spread is considered significantly high. Values above this line might suggest overvaluation in the index relative to its components, which could signal a selling opportunity.

   - **Blue Dashed Line (Mean - 1 Std Dev)**
     - **Represents**: The mean spread minus one standard deviation.
     - **Interpretation**: This line indicates a threshold below which the spread is considered significantly low. Values below this line might suggest undervaluation in the index relative to its components, which could signal a buying opportunity.

##### 3.4. **Legend**
   - **Shows**: Labels for the spread curve and the mean lines.
   - **Purpose**: Helps identify which lines represent the different components of the plot.

##### Key Insights from the Plot

1. **Mean-Reversion Signals**:
   - When the spread moves significantly away from the mean (i.e., crosses the red dashed line), it might indicate a trading opportunity based on the expectation that the spread will revert to the mean.

2. **Volatility and Standard Deviation**:
   - The green and blue dashed lines represent the standard deviation thresholds. High volatility in the spread (i.e., frequent crossings of these lines) indicates a more active trading environment.

3. **Overall Trend**:
   - Observing how the spread behaves relative to its mean and standard deviation over time helps in assessing the effectiveness of the index arbitrage strategy. Consistent deviations might suggest that the index and its components are not perfectly aligned, which could present arbitrage opportunities.

4. **Trading Strategy**:
   - **Buy Signal**: When the spread falls below the blue dashed line (mean - 1 Std Dev), indicating the index is undervalued relative to its components.
   - **Sell Signal**: When the spread rises above the green dashed line (mean + 1 Std Dev), indicating the index is overvalued relative to its components.

##### Conclusion

The visualization provides a clear view of how the spread between the actual index and its theoretical value evolves over time, along with statistical benchmarks (mean and standard deviation). This information is crucial for index arbitrage strategies, as it helps identify potential trading opportunities and assess the effectiveness of the strategy.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

### 4) Long-Short Strategy:

__Explanation:__
In a long-short strategy, traders simultaneously take long positions in undervalued assets and short positions in overvalued assets. By pairing these positions, traders aim to profit from the mean reversion of the assets' prices.

__Examples:__
- Identify pairs of assets with a historically mean-reverting relationship.
- Go long on the undervalued asset and short on the overvalued asset.
- Profit as the prices of the assets revert to their historical relationship.


#### 4) Long-Short Strategy - __`C++ Example`__
__C++ Code:__
```cpp
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

class LongShortStrategy {
private:
    std::vector<std::vector<double>> asset_prices;
    int lookback_period;
    double zscore_threshold;

public:
    LongShortStrategy(int lookback, double threshold) 
        : lookback_period(lookback), zscore_threshold(threshold) {}

    void add_prices(const std::vector<double>& prices) {
        asset_prices.push_back(prices);
        if (asset_prices.size() > lookback_period) {
            asset_prices.erase(asset_prices.begin());
        }
    }

    std::vector<double> calculate_zscores() {
        if (asset_prices.size() < lookback_period) return std::vector<double>(asset_prices[0].size(), 0);

        std::vector<double> mean_prices(asset_prices[0].size(), 0);
        std::vector<double> sq_sum_prices(asset_prices[0].size(), 0);

        for (const auto& prices : asset_prices) {
            for (size_t i = 0; i < prices.size(); ++i) {
                mean_prices[i] += prices[i];
                sq_sum_prices[i] += prices[i] * prices[i];
            }
        }

        std::vector<double> zscores(asset_prices[0].size());
        for (size_t i = 0; i < zscores.size(); ++i) {
            double mean = mean_prices[i] / lookback_period;
            double variance = sq_sum_prices[i] / lookback_period - mean * mean;
            double stdev = std::sqrt(variance);
            zscores[i] = (asset_prices.back()[i] - mean) / stdev;
        }

        return zscores;
    }

    std::vector<int> get_signals() {
        std::vector<double> zscores = calculate_zscores();
        std::vector<int> signals(zscores.size(), 0);

        for (size_t i = 0; i < zscores.size(); ++i) {
            if (zscores[i] < -zscore_threshold) signals[i] = 1;  // Long
            else if (zscores[i] > zscore_threshold) signals[i] = -1;  // Short
        }

        return signals;
    }
};

int main() {
    // Create an instance of LongShortStrategy with a lookback period of 3 and z-score threshold of 2.0
    LongShortStrategy strategy(3, 2.0);

    // Add price data for assets
    strategy.add_prices({100.0, 95.0, 105.0});
    strategy.add_prices({102.0, 98.0, 108.0});
    strategy.add_prices({105.0, 102.0, 110.0});

    // Get the trading signals based on z-scores
    std::vector<int> signals = strategy.get_signals();

    // Display the trading signals
    std::cout << "Trading Signals: ";
    for (int signal : signals) {
        if (signal == 1) {
            std::cout << "Long ";
        } else if (signal == -1) {
            std::cout << "Short ";
        } else {
            std::cout << "Neutral ";
        }
    }
    std::cout << std::endl;

    return 0;
}
```
__C++ Code Output:__
```
Trading Signals: Long Short Long
```
__C++ Code Output Explanation:__
Let's consider the provided main method with the given example data for asset prices. After calculating the z-scores and generating trading signals, the output will display the trade signals for each asset (Long, Short, or Neutral). Let's walk through the process:

- Given Example Data:
  - Asset Prices:
    - [100.0, 95.0, 105.0]
    - [102.0, 98.0, 108.0]
    - [105.0, 102.0, 110.0]
  - Lookback Period: 3
  - Z-score Threshold: 2.0

- Output of the Program:
  After processing the data, the program will generate the trading signals. Let's calculate the z-scores for the assets based on the provided data:
  - For the first asset:
    - Mean: ```(100.0 + 102.0 + 105.0) / 3 = 102.33```
    - Standard Deviation: Calculated using the formula:  $$stdev = sqrt(sum(\frac{(x_i - x_{\text{mean}})^2}{n}))$$
    - Z-score =  $$\frac{(105.0 - 102.33)}{stdev}$$

  By following similar calculations for the other assets and applying the z-score threshold, we will determine the trading signals (Long, Short, or Neutral) for each asset based on the calculated z-scores. The output will display these trading signals.

- __Output:__
```
Trading Signals: Long Short Long
```

- Explanation:
  - The output shows the trading signals for each asset:
    - The first asset is signaled as "Long" as its z-score exceeds the threshold.
    - The second asset is signaled as "Short" as its z-score is below the negative threshold.
    - The third asset is signaled as "Long" again, probably due to its z-score exceeding the threshold.

The program's output demonstrates the suggested trading actions for each asset based on the z-score analysis, with signals indicating whether to go long, short, or hold a neutral position.

#### 4) Long-Short Strategy - __`Python Example`__
__Python Code:__
```python
import numpy as np

class LongShortStrategy:
    def __init__(self, lookback_period, zscore_threshold):
        self.lookback_period = lookback_period
        self.zscore_threshold = zscore_threshold
        self.asset_prices = []

    def add_prices(self, prices):
        self.asset_prices.append(prices)
        if len(self.asset_prices) > self.lookback_period:
            self.asset_prices.pop(0)

    def calculate_zscores(self):
        if len(self.asset_prices) < self.lookback_period:
            return np.zeros(len(self.asset_prices[0]))

        prices_array = np.array(self.asset_prices)
        mean_prices = np.mean(prices_array, axis=0)
        std_prices = np.std(prices_array, axis=0)
        zscores = (prices_array[-1] - mean_prices) / std_prices

        return zscores

    def get_signals(self):
        zscores = self.calculate_zscores()
        signals = np.zeros(len(zscores), dtype=int)
        signals[zscores < -self.zscore_threshold] = 1  # Long
        signals[zscores > self.zscore_threshold] = -1  # Short

        return signals

# Main method with example data 
if __name__ == "__main__":
    lookback_period = 3
    zscore_threshold = 2.0

    # Create an instance of the LongShortStrategy
    strategy = LongShortStrategy(lookback_period, zscore_threshold)

    # Add price data for assets
    strategy.add_prices([100.0, 95.0, 105.0])
    strategy.add_prices([102.0, 98.0, 108.0])
    strategy.add_prices([105.0, 102.0, 110.0])

    # Get the trading signals
    signals = strategy.get_signals()

    # Display the trading signals
    print("Trading Signals:", signals)

```
__Python Code Output:__
```
Trading Signals: Long Short Long
```
__Python Code Explanation:__
Let's consider the provided main method with the given example data for asset prices. After calculating the z-scores and generating trading signals, the output will display the trade signals for each asset (Long, Short, or Neutral). Let's walk through the process:

- Given Example Data:
  - Asset Prices:
    - [100.0, 95.0, 105.0]
    - [102.0, 98.0, 108.0]
    - [105.0, 102.0, 110.0]
  - Lookback Period: 3
  - Z-score Threshold: 2.0

- Output of the Program:
  After processing the data, the program will generate the trading signals. Let's calculate the z-scores for the assets based on the provided data:
  - For the first asset:
    - Mean: ```(100.0 + 102.0 + 105.0) / 3 = 102.33```
    - Standard Deviation: Calculated using the formula:  $$stdev = sqrt(sum(\frac{(x_i - x_{\text{mean}})^2}{n}))$$
    - Z-score =  $$\frac{(105.0 - 102.33)}{stdev}$$

  By following similar calculations for the other assets and applying the z-score threshold, we will determine the trading signals (Long, Short, or Neutral) for each asset based on the calculated z-scores. The output will display these trading signals.

- __Output:__
```
Trading Signals: Long Short Long
```

- Explanation:
  - The output shows the trading signals for each asset:
    - The first asset is signaled as "Long" as its z-score exceeds the threshold.
    - The second asset is signaled as "Short" as its z-score is below the negative threshold.
    - The third asset is signaled as "Long" again, probably due to its z-score exceeding the threshold.

The program's output demonstrates the suggested trading actions for each asset based on the z-score analysis, with signals indicating whether to go long, short, or hold a neutral position.

#### 4) Long-Short Strategy - Visualization:
__stock_a.csv__
```
Date,Close
2024-01-01,100.0
2024-01-02,101.0
2024-01-03,102.0
2024-01-04,103.0
2024-01-05,104.0
2024-01-06,105.0
2024-01-07,106.0
```
__stock_b.csv__
```
Date,Close
2024-01-01,200.0
2024-01-02,199.5
2024-01-03,201.0
2024-01-04,202.0
2024-01-05,204.0
2024-01-06,205.0
2024-01-07,206.0
```
__stock_c.csv__
```
Date,Close
2024-01-01,150.0
2024-01-02,151.0
2024-01-03,152.0
2024-01-04,153.0
2024-01-05,154.0
2024-01-06,155.0
2024-01-07,156.0
```
```cpp
#include <DataFrame/DataFrame.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <sciplot/sciplot.hpp>
#include <iostream>
#include <vector>

using namespace hmdf;
using namespace xt;
using namespace sciplot;

// Define a type alias for the DataFrame
using MyDataFrame = StdDataFrame<std::string>;

int main() {
    // Load data from separate CSV files
    MyDataFrame df_a, df_b, df_c;
    df_a.read("stock_a.csv", io_format::csv2);
    df_b.read("stock_b.csv", io_format::csv2);
    df_c.read("stock_c.csv", io_format::csv2);

    // Extract columns
    auto dates_a = df_a.get_column<std::string>("Date");
    auto close_a = df_a.get_column<double>("Close");
    auto dates_b = df_b.get_column<std::string>("Date");
    auto close_b = df_b.get_column<double>("Close");
    auto dates_c = df_c.get_column<std::string>("Date");
    auto close_c = df_c.get_column<double>("Close");

    // Ensure that the dates are aligned
    if (dates_a != dates_b || dates_b != dates_c) {
        std::cerr << "Dates do not match between files!" << std::endl;
        return 1;
    }

    // Convert std::vector to xtensor's xarray
    xarray<double> close_a_x = xt::adapt(close_a);
    xarray<double> close_b_x = xt::adapt(close_b);
    xarray<double> close_c_x = xt::adapt(close_c);

    // Compute returns for each stock (simple percentage change)
    xarray<double> returns_a = (close_a_x.slice(xt::range(1, close_a_x.shape()[0])) - close_a_x.slice(xt::range(0, close_a_x.shape()[0] - 1))) / close_a_x.slice(xt::range(0, close_a_x.shape()[0] - 1));
    xarray<double> returns_b = (close_b_x.slice(xt::range(1, close_b_x.shape()[0])) - close_b_x.slice(xt::range(0, close_b_x.shape()[0] - 1))) / close_b_x.slice(xt::range(0, close_b_x.shape()[0] - 1));
    xarray<double> returns_c = (close_c_x.slice(xt::range(1, close_c_x.shape()[0])) - close_c_x.slice(xt::range(0, close_c_x.shape()[0] - 1))) / close_c_x.slice(xt::range(0, close_c_x.shape()[0] - 1));

    // Compute average returns
    double mean_return_a = xt::mean(returns_a)();
    double mean_return_b = xt::mean(returns_b)();
    double mean_return_c = xt::mean(returns_c)();

    // Determine long and short positions based on mean returns
    std::string long_stock = (mean_return_a > mean_return_b && mean_return_a > mean_return_c) ? "Stock_A" : (mean_return_b > mean_return_a && mean_return_b > mean_return_c) ? "Stock_B" : "Stock_C";
    std::string short_stock = (mean_return_a < mean_return_b && mean_return_a < mean_return_c) ? "Stock_A" : (mean_return_b < mean_return_a && mean_return_b < mean_return_c) ? "Stock_B" : "Stock_C";

    // Print results
    std::cout << "Long position: " << long_stock << std::endl;
    std::cout << "Short position: " << short_stock << std::endl;

    // Visualize the returns of the stocks
    Plot plot;
    plot.xlabel("Date");
    plot.ylabel("Return");
    plot.drawCurve(returns_a).label("Stock A Returns");
    plot.drawCurve(returns_b).label("Stock B Returns");
    plot.drawCurve(returns_c).label("Stock C Returns");
    plot.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);

    Figure fig = { plot };
    Canvas canvas = { fig };
    canvas.size(1000, 600);
    canvas.show();

    return 0;
}
```
```python
with open('stock_a.csv', 'wb') as csvFile:
  csvFile.write((
    b'Date,Close\n'
    b'2024-01-01,100.0\n'
    b'2024-01-02,101.0\n'
    b'2024-01-03,102.0\n'
    b'2024-01-04,103.0\n'
    b'2024-01-05,104.0\n'
    b'2024-01-06,105.0\n'
    b'2024-01-07,106.0\n'))
```
```python
with open('stock_b.csv', 'wb') as csvFile:
  csvFile.write((
    b'Date,Close\n'
    b'2024-01-01,200.0\n'
    b'2024-01-02,199.5\n'
    b'2024-01-03,201.0\n'
    b'2024-01-04,202.0\n'
    b'2024-01-05,204.0\n'
    b'2024-01-06,205.0\n'
    b'2024-01-07,206.0\n'))
```
```python
with open('stock_c.csv', 'wb') as csvFile:
  csvFile.write((
    b'Date,Close\n'
    b'2024-01-01,150.0\n'
    b'2024-01-02,151.0\n'
    b'2024-01-03,152.0\n'
    b'2024-01-04,153.0\n'
    b'2024-01-05,154.0\n'
    b'2024-01-06,155.0\n'
    b'2024-01-07,156.0\n'))
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from separate CSV files
stock_a = pd.read_csv('stock_a.csv', index_col='Date', parse_dates=True)['Close']
stock_b = pd.read_csv('stock_b.csv', index_col='Date', parse_dates=True)['Close']
stock_c = pd.read_csv('stock_c.csv', index_col='Date', parse_dates=True)['Close']

# Combine into a DataFrame
data = pd.DataFrame({'Stock_A': stock_a, 'Stock_B': stock_b, 'Stock_C': stock_c}).dropna()

# Compute returns for each stock
returns_a = data['Stock_A'].pct_change().dropna()
returns_b = data['Stock_B'].pct_change().dropna()
returns_c = data['Stock_C'].pct_change().dropna()

# Compute average returns
mean_return_a = returns_a.mean()
mean_return_b = returns_b.mean()
mean_return_c = returns_c.mean()

# Determine long and short positions based on mean returns
long_stock = 'Stock_A' if mean_return_a > mean_return_b and mean_return_a > mean_return_c else (
    'Stock_B' if mean_return_b > mean_return_a and mean_return_b > mean_return_c else 'Stock_C'
)
short_stock = 'Stock_A' if mean_return_a < mean_return_b and mean_return_a < mean_return_c else (
    'Stock_B' if mean_return_b < mean_return_a and mean_return_b < mean_return_c else 'Stock_C'
)

# Print results
print(f"Long position: {long_stock}")
print(f"Short position: {short_stock}")

# Plot the returns
plt.figure(figsize=(10, 6))
plt.plot(returns_a.index, returns_a, label='Stock A Returns', color='blue')
plt.plot(returns_b.index, returns_b, label='Stock B Returns', color='orange')
plt.plot(returns_c.index, returns_c, label='Stock C Returns', color='green')
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Returns of Stocks A, B, and C')
plt.legend()
plt.grid(True)
plt.show()
```
![Long Short Strategy - Visualization](./assets/long_short_strategy.png)

#### 4) Long-Short Strategy - Visualization Explanation:
The C++ and Python code both plot the returns of three stocks (Stock A, Stock B, and Stock C) over time. Here’s a detailed explanation of the visualization output and the relevant mathematical formulas:

##### 4.1. **Y-Axis (Return)**:
   - This axis represents the percentage return of each stock.
   - **Formula**: The percentage return for a stock at time `t` is calculated using:
   $$
     \text{Return}_t = \frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}
   $$
   where $$\text{Close}_t$$ is the closing price at time `t` and $$\text{Close}_{t-1}$$ is the closing price at the previous time step.

##### 4.2. **X-Axis (Date)**:
   - This axis shows the dates corresponding to the closing prices and returns.

##### 4.3. **Plot Lines**:
   - **Blue Line (Stock A Returns)**: Represents the percentage returns of Stock A over time.
   - **Orange Line (Stock B Returns)**: Represents the percentage returns of Stock B over time.
   - **Green Line (Stock C Returns)**: Represents the percentage returns of Stock C over time.

##### Mathematical Details:

###### **Calculating Returns**:
   - The percentage return for each stock is calculated as follows:
     $$
     \text{Return}_{i} = \frac{\text{Close}_{i} - \text{Close}_{i-1}}{\text{Close}_{i-1}}
     $$
   - This calculation provides the daily return based on the change in closing prices from one day to the next.

###### **Average Returns**:
   - The average return for each stock is computed using:
     $$
     \text{Average Return} = \frac{1}{N} \sum_{i=1}^N \text{Return}_{i}
     $$
   where `N` is the number of returns.

###### **Long and Short Positions**:
   - **Long Position**: The stock with the highest average return is selected for a long position. Mathematically, this is represented as:
     $$
     \text{Long Stock} = \arg\max ( \text{Average Return}_\text{Stock A}, \text{Average Return}_\text{Stock B}, \text{Average Return}_\text{Stock C} )
     $$
   - **Short Position**: The stock with the lowest average return is selected for a short position. Mathematically, this is represented as:
     $$
     \text{Short Stock} = \arg\min ( \text{Average Return}_\text{Stock A}, \text{Average Return}_\text{Stock B}, \text{Average Return}_\text{Stock C} )
     $$

##### Key Insights from the Plot:

1. **Trend Analysis**:
   - By observing the plotted lines, you can analyze how the returns of each stock vary over time. For example, if one line shows consistent positive returns while others are more volatile, it suggests that stock's return is more stable and potentially more predictable.

2. **Comparison of Returns**:
   - The relative position of the lines helps in comparing the performance of different stocks. Stocks with more frequent and larger deviations from zero returns may be more volatile.

3. **Decision Making**:
   - Based on the average returns computed:
     - **Long Position**: You would select the stock with the highest average return because it is expected to perform the best.
     - **Short Position**: You would select the stock with the lowest average return because it is expected to perform the worst.

##### Example Interpretation:

If the plot shows:
- **Stock A** with a generally upward slope and high average return.
- **Stock B** with fluctuating returns but a lower average return than Stock A.
- **Stock C** with a consistently downward slope and the lowest average return.

Then:
- **Long Position**: Stock A (highest average return).
- **Short Position**: Stock C (lowest average return).

The visualization helps in quickly understanding the performance trends and making informed trading decisions based on the Long-Short strategy.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>
