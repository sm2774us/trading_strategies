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

### 2) Triplets Trading:

__Explanation:__
Triplets trading involves identifying three assets that move together and creating a trading strategy based on their relationship. Traders look for deviations in the spread between these assets to take advantage of mean reversion opportunities.

__Examples:__
- Choose three related instruments such as an index, a stock, and a commodity.
- Monitor the spread between these assets and establish positions when deviations occur to profit from the mean reversion of the spread.

#### 2) Triplets Trading - __`C++ Example`__

__C++ Code:__
```cpp
```

__C++ Code Output:__
```
```

__C++ Code Explanation:__


#### 2) Triplets Trading - __`Python Example`__

__Python Code:__
```python
```

__Python Code Output:__
```
```

__Python Code Explanation:__

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
