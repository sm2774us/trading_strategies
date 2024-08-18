# Asset Classes - Mathematics
Let’s dive into the detailed mathematics and statistics involved in trading each of the asset categories, including necessary formulas and proofs where relevant.

### 1) **Foreign Exchange (FX)**

#### **Key Concepts:**
1. **Exchange Rates**:
   - **Definition**: The price of one currency in terms of another. If $E_{USD/EUR}$ represents the exchange rate from USD to EUR, then $1 \, USD = E_{USD/EUR} \times \text{EUR}$.
   - **Formula**: $$E_{USD/EUR} = \frac{1}{E_{EUR/USD}}$$.

2. **Interest Rate Parity (IRP)**:
   - **Concept**: IRP states that the difference between the forward exchange rate $F$ and the spot exchange rate $S$ is a function of the interest rate differential between two currencies.
   - **Formula**: $$\frac{F}{S} = \frac{1 + r_d}{1 + r_f}$$, where $r_d$ and $r_f$ are the domestic and foreign interest rates, respectively.

   **Proof**:
   Let’s consider two investment strategies:
   - **Strategy 1**: Invest $1$ unit of domestic currency at the domestic interest rate $r_d$.
   - **Strategy 2**: Convert $1$ unit of domestic currency to foreign currency at the spot rate $S$, invest it at the foreign interest rate $r_f$, and convert it back to domestic currency at the forward rate $F$.

   Equating the final returns from both strategies gives the IRP formula above.

3. **Volatility**:
   - **GARCH Model**: The Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model is used to model and forecast time-series volatility, which is crucial in FX trading.
   - **Formula**: 
     $$
     \sigma_t^2 = \alpha_0 + \sum_{i=1}^q \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2
     $$
     where $\sigma_t^2$ is the conditional variance, $\epsilon_t$ are past residuals, and $\alpha_0, \alpha_i, \beta_j$ are model parameters.

#### **Additional Mathematical Tools:**
- **Stochastic Calculus**:
   - **Itô’s Lemma**: If $X_t$ follows a stochastic process $$dX_t = \mu_t dt + \sigma_t dW_t$$, then for a function $f(X_t, t)$:
     $$
     df(X_t, t) = \left( \frac{\partial f}{\partial t} + \mu_t \frac{\partial f}{\partial X_t} + \frac{1}{2} \sigma_t^2 \frac{\partial^2 f}{\partial X_t^2} \right) dt + \sigma_t \frac{\partial f}{\partial X_t} dW_t
     $$
   - This is particularly used in deriving option pricing formulas in FX markets.

### 2) **Equities**

#### **Key Concepts:**
1. **CAPM (Capital Asset Pricing Model)**:
   - **Formula**: 
     $$
     E(R_i) = R_f + \beta_i (E(R_m) - R_f)
     $$
     where $E(R_i)$ is the expected return of asset $i$, $R_f$ is the risk-free rate, $\beta_i$ is the asset’s beta, and $E(R_m)$ is the expected market return.
   
   **Proof**:
   - Derived from the equilibrium conditions of the market where the expected return of an asset is proportional to its beta with respect to the market portfolio.

2. **Dividend Discount Model (DDM)**:
   - **Formula**: 
     $$
     P_0 = \frac{D_1}{r - g}
     $$
     where $P_0$ is the present stock price, $D_1$ is the dividend next period, $r$ is the required rate of return, and $g$ is the growth rate of dividends.
   
   **Proof**:
   - Based on the present value of a perpetuity with growth, where dividends are assumed to grow at a constant rate $g$.

#### **Mathematical Tools:**
- **Regression Analysis**:
   - Used to estimate the beta of a stock by regressing the stock’s returns against market returns:
     $$
     R_i = \alpha + \beta R_m + \epsilon
     $$
     where $R_i$ is the return of stock $i$, $R_m$ is the market return, and $\epsilon$ is the error term.

- **Black-Scholes Model for Options**:
   - **Formula**: 
     $$
     C(S,t) = S_0 N(d_1) - X e^{-rt} N(d_2)
     $$
     where
     $$
     d_1 = \frac{\ln(\frac{S_0}{X}) + (r + \frac{\sigma^2}{2})t}{\sigma \sqrt{t}}
     $$,
     $$
     d_2 = d_1 - \sigma \sqrt{t}
     $$,
     $S_0$ is the current stock price, $X$ is the strike price, $r$ is the risk-free rate, $\sigma$ is the volatility, and $N(d)$ is the cumulative distribution function of the standard normal distribution.

### 3) **Futures**

#### **Key Concepts:**
1. **Cost of Carry Model**:
   - **Formula**: 
     $$
     F = S e^{(r + c - y)t}
     $$
     where $F$ is the futures price, $S$ is the spot price, $r$ is the risk-free rate, $c$ is the storage cost, $y$ is the yield on the asset, and $t$ is the time to maturity.
   
   **Proof**:
   - Based on the idea that the cost of carrying an asset forward should equal the futures price minus the spot price, accounting for the time value of money, storage costs, and yields.

2. **Basis and Hedging**:
   - **Basis**: 
     $$
     \text{Basis} = S - F
     $$
   - The basis represents the difference between the spot price and the futures price, and it’s critical in hedging strategies to assess the effectiveness of a hedge.

#### **Mathematical Tools:**
- **Value at Risk (VaR)**:
   - **Formula**: 
     $$
     \text{VaR}_{\alpha} = \mu - z_{\alpha} \sigma
     $$
     where $\mu$ is the mean of the portfolio returns, $\sigma$ is the standard deviation, and $z_{\alpha}$ is the z-score corresponding to the confidence level $\alpha$.

- **Spread Analysis**:
   - **Formula**:
     $$
     \text{Spread} = F_1 - F_2
     $$
     where $F_1$ and $F_2$ are futures prices of contracts with different maturities.

### 4) **Treasuries**

#### **Key Concepts:**
1. **Yield to Maturity (YTM)**:
   - **Formula**: 
     $$
     P = \sum_{t=1}^n \frac{C}{(1 + \text{YTM})^t} + \frac{F}{(1 + \text{YTM})^n}
     $$
     where $P$ is the bond price, $C$ is the coupon payment, $F$ is the face value, and $n$ is the number of periods.
   
   **Proof**:
   - The bond price equals the present value of its future cash flows, discounted at the YTM.

2. **Duration and Convexity**:
   - **Duration Formula**:
     $$
     D = \sum_{t=1}^n \frac{t \times C_t}{(1 + r)^t} \times \frac{1}{P}
     $$
     where $C_t$ is the cash flow at time $t$, $r$ is the yield, and $P$ is the bond price.
   - **Convexity Formula**:
     $$
     C = \frac{1}{P} \sum_{t=1}^n \frac{C_t \times t \times (t + 1)}{(1 + r)^{t+2}}
     $$

#### **Mathematical Tools:**
- **Nelson-Siegel Model**:
   - **Formula**:
     $$
     y(\tau) = \beta_0 + \beta_1 \frac{1 - e^{-\lambda \tau}}{\lambda \tau} + \beta_2 \left( \frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau} \right)
     $$
     where $y(\tau)$ is the yield for maturity $\tau$, and $\beta_0, \beta_1, \beta_2, \lambda$ are model parameters.

### 5) **Interest Rates**

#### **Key Concepts:**
1. **Libor and Swap Rates**:
   - **Forward Rate Formula**:
     $$
     F(t_1, t_2) = \frac{(1 + r(t_2) \times (t_2 - t_0))}{(1 + r(t_1) \times (t_1 - t_0))} - 1
     $$
     where $r(t)$ is the spot rate at time $t$.

2. **Affine Term Structure Models (ATSM)**:
   - **Basic Formula**:
     $$
     P(t,T) = e^{A(t,T) + B(t,T) r(t)}
     $$
     where $P(t,T)$ is the price of a zero-coupon bond, $r(t)$ is the short rate, and $A(t,T)$ and $B(t,T)$ are functions determined by the specific model (e.g., __Vasicek, Cox-Ingersoll-Ross__).

#### **Mathematical Tools:**
- **Vasicek Model**:
   - **SDE**: 
     $$
     dr_t = \alpha (\mu - r_t) dt + \sigma dW_t
     $$
     where $\alpha$ is the speed of mean reversion, $\mu$ is the long-term mean, and $\sigma$ is the volatility.

### 6) **Options**

#### **Key Concepts:**
1. **The Greeks**:
   - **Delta**: Sensitivity of option price to changes in the underlying asset price.
     $$
     \Delta = \frac{\partial C}{\partial S}
     $$
   - **Gamma**: Sensitivity of Delta to changes in the underlying asset price.
     $$
     \Gamma = \frac{\partial \Delta}{\partial S} = \frac{\partial^2 C}{\partial S^2}
     $$
   - **Theta**: Sensitivity of option price to the passage of time.
     $$
     \Theta = \frac{\partial C}{\partial t}
     $$
   - **Vega**: Sensitivity of option price to changes in volatility.
     $$
     \nu = \frac{\partial C}{\partial \sigma}
     $$

2. **Volatility Surface**:
   - **Implied Volatility**: Volatility inferred from the market price of an option.
   - **Smile and Skew**: Patterns observed in implied volatility for different strike prices.

#### **Mathematical Tools:**
- **Black-Scholes PDE**:
   - **Formula**:
     $$
     \frac{\partial C}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} + r S \frac{\partial C}{\partial S} - r C = 0
     $$
   - Derived using Itô’s Lemma and the construction of a risk-neutral portfolio.

- **Binomial Tree Model**:
   - **Formula**:
     $$
     C = \frac{1}{(1 + r)^n} \left( \sum_{i=0}^n \binom{n}{i} p^i (1 - p)^{n-i} C(S_i) \right)
     $$
     where $p$ is the probability of an up move, $C(S_i)$ is the option value at node $i$.

### 7) **Fixed Income**

#### **Key Concepts:**
1. **Credit Spread**:
   - **Formula**:
     $$
     \text{Spread} = YTM_{corporate} - YTM_{treasury}
     $$
     where the spread reflects the additional yield required by investors for taking on more credit risk.

2. **Bond Immunization**:
   - **Concept**: Strategy to make a bond portfolio immune to changes in interest rates by matching the portfolio's duration to the investment horizon.

#### **Mathematical Tools:**
- **Optimization Models**:
   - **Linear Programming**: For constructing optimal bond portfolios subject to constraints (e.g., duration, convexity).
   - **Duration Matching**: Aligning the duration of assets and liabilities to protect against interest rate movements.

### 8) **Derivatives**

#### **Key Concepts:**
1. **Payoff Diagrams**:
   - **Example**: For a call option, the payoff at expiration is $\max(S_T - X, 0)$, where $S_T$ is the underlying price at maturity and $X$ is the strike price.

2. **Hedging Strategies**:
   - **Delta Hedging**: Continuously adjusting the quantity of the underlying asset to maintain a delta-neutral position.

#### **Mathematical Tools:**
- **Finite Difference Methods**:
   - **Example**: Solving the Black-Scholes PDE numerically using discretization methods.
   - **Formula**:
     $$
     C_{i,j} = \frac{\Delta t}{2} \left( \sigma^2 S_i^2 \frac{C_{i+1,j} - 2C_{i,j} + C_{i-1,j}}{(\Delta S)^2} + r S_i \frac{C_{i+1,j} - C_{i-1,j}}{2 \Delta S} \right) + C_{i,j+1}
     $$

### 9) **Cryptocurrencies (Crypto)**

#### **Key Concepts:**
1. **Blockchain Analytics**:
   - **Transaction Analysis**: Using blockchain data to track large movements of cryptocurrencies, which can signal market shifts.
   - **Network Metrics**: Metrics like hash rate, transaction volume, and active addresses can be analyzed statistically to gauge network health.

2. **Market Sentiment Analysis**:
   - **Sentiment Score**: Aggregating sentiment from social media and news sources to predict price movements.

#### **Mathematical Tools:**
- **Time Series Analysis**:
   - **ARIMA Models**: For forecasting crypto prices based on historical data.
   - **Formula**: 
     $$
     y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \sum_{i=1}^q \theta_i \epsilon_{t-i} + \epsilon_t
     $$
     where $y_t$ is the price, $\phi_i$ are the autoregressive coefficients, $\theta_i$ are the moving average coefficients, and $\epsilon_t$ is the error term.

- **Machine Learning Models**:
   - **Neural Networks**: For predicting price movements using large datasets, incorporating factors like sentiment, volume, and technical indicators.
   - **Feature Engineering**: Extracting relevant features from blockchain and market data for predictive modeling.

This detailed exploration touches on the most essential mathematical and statistical concepts used in trading across these asset categories. The formulas and tools mentioned here form the foundation of quantitative finance, helping traders and analysts to make informed decisions, manage risk, and optimize their strategies.