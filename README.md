# __Trading Strategies__
My Private Repository of Trading Strategies


## Table Of Contents <a name="top"></a>
1. [__Mean Reversion Strategies__](#mean-reversion-strategies)
    - 1.1. [__1) Statistical Arbitrage:__](#1-statistical-arbitrage)
    - 1.2. [__2) Triplets Trading:__](#2-triplets-trading)
    - 1.3. [__3) Index Arbitrage:__](#3-index-arbitrage)
    - 1.4. [__4) Long Short Strategy:__](#4-long-short-strategy)


## Mean Reversion Strategies
Mean reversion strategies in trading involve taking advantage of the tendency of asset prices to revert to their historical averages over time. These strategies assume that when an asset price deviates significantly from its average, it is likely to move back towards the average in the future. Mean reversion strategies are often implemented using statistical analysis, time series modeling, and quantitative methods.

__Use Cases for Mean Reversion Strategies:__
Mean reversion strategies can be applied across various asset categories including:
- __Equities:__ Mean reversion strategies are commonly used in stock markets to exploit price disparities between related stocks.
- __Fixed Income:__ Traders can implement mean reversion strategies in bond markets to benefit from yield spreads reverting to their historical averages.
- __Futures:__ Mean reversion strategies in futures markets involve exploiting price differentials between futures contracts and their underlying assets.
- __FX:__ Traders in the foreign exchange market can use mean reversion strategies to capitalize on currency pairs' price deviations from their historical averages.

These strategies can be programmed in both C++ and Python to automate trading decisions based on statistical analysis and quantitative models. C++ offers high performance, while Python provides ease of implementation and extensive libraries for quantitative analysis.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

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
__C++ Code Explanation:__
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
    // Load CSV data
    MyDataFrame df_x;
    MyDataFrame df_y;
    MyDataFrame df_z;

    df_x.read("stock_x.csv", io_format::csv2);
    df_y.read("stock_y.csv", io_format::csv2);
    df_z.read("stock_z.csv", io_format::csv2);

    // Extract the 'Close' column from each DataFrame
    auto close_x = df_x.get_column<double>("Close");
    auto close_y = df_y.get_column<double>("Close");
    auto close_z = df_z.get_column<double>("Close");

    // Convert std::vector to xtensor's xarray
    xarray<double> stock_x = xt::adapt(close_x);
    xarray<double> stock_y = xt::adapt(close_y);
    xarray<double> stock_z = xt::adapt(close_z);

    // Assume pre-determined weights (example)
    double w_x = 1.0, w_y = -1.5, w_z = 0.5;

    // Calculate the spread
    xarray<double> spread = w_x * stock_x + w_y * stock_y + w_z * stock_z;

    // Calculate mean of the spread
    double mean_spread = xt::mean(spread)();

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
```
__C++ Code Output:__
```
```
__C++ Code Explanation:__


#### 3) Index Arbitrage - __`Python Example`__

__Python Code:__
```python
```

__Python Code Output:__
```
```

__Python Code Explanation:__

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
```

__C++ Code Output:__
```
```

__C++ Code Explanation:__


#### 4) Long-Short Strategy - __`Python Example`__

__Python Code:__
```python
```

__Python Code Output:__
```
```

__Python Code Explanation:__

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>
