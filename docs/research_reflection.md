# Research Reflection: Forecasting S&P 500 Volatility with a Kalman Filter

*Kaiki Ikeda · October 2026*

## What I set out to test

At FPT I built structural time-series models for car sales: a Kalman filter separating a hidden trend and seasonality from noisy monthly data. Volatility looked like the same problem in a different setting. Nobody observes it; you only see returns, which are a very noisy reading of it. So I wrote down one falsifiable claim before writing any code: *a stochastic-volatility model estimated with a Kalman filter will forecast next month's S&P 500 realized volatility better than GARCH, HAR, and the VIX.*

I set the rules first: expanding-window walk-forward from 2005, parameters refit every 21 trading days with only past data, every model scored on the same 5,443 daily forecasts, and the trading signal fixed in advance to use my SV model.

## What I found

The claim was wrong. The SV model ranked seventh of eight on QLIKE, ahead of only the naive "last month's volatility" forecast, and it was worse than HAR at the 10% level (Diebold–Mariano p = 0.07). GJR-GARCH and HAR with the VIX as a regressor tied for best.

I think a rejected hypothesis is a more useful result than a weak confirmation would have been, because the reasons it failed were specific and I could measure each of them.

## Why the elegant model lost

**One day's return is a very weak signal about volatility.** In the state-space form, the observation is log r² and its noise has variance π²/2 ≈ 4.9, while the hidden log-variance itself has a variance of only about 1.1. The steady-state Kalman gain works out to 0.065: each new return moves the volatility estimate by about 6.5% of its surprise. That is the right amount of caution on an ordinary day and far too little when the market breaks. Every point of the SV model's deficit came from forecasts made just before calm turned into crisis, in the eight worst months (including February 2020, January and October 2018, and the month before the 2010 flash crash). Outside those months it was slightly better than HAR.

**The log transform throws away the sign of the return.** log r² treats a 3% rally and a 3% crash identically. The best single model was GJR-GARCH, whose only difference from GARCH is that it reacts more to down days. The leverage effect I had written about in my primer showed up directly in the out-of-sample ranking, and my model was structurally blind to it.

**Being unbiased is not the same as being useful.** Apart from the naive forecast, my SV model had the smallest average bias of any model (−0.15 vol points). The others, including the VIX, were biased high. Under QLIKE, under-predicting volatility is punished much more than over-predicting it, and an unbiased forecaster under-predicts half the time, including right before crises. Before this project I would have assumed an unbiased forecast was automatically a better one. Choosing the loss function is a modeling decision: QLIKE is the loss of someone who is short options, and it rewards a margin of safety.

## The bug a unit test caught

The standard way to handle log(0) on days the index doesn't move is a small offset, log(r² + κ). I used κ = 0.005, which seemed negligible. I wrote a test that simulates SV returns with known parameters and checks that the estimator recovers them. It failed: true σ²_η = 0.0225, estimate 0.008.

The offset squeezes the long left tail of log ε², so the real noise variance falls well below the π²/2 the model assumes. The filter then explains the "missing" noise by assuming volatility barely moves. On the real data, removing the offset (treating the three zero-return days as missing observations, which the Kalman filter handles exactly) raised the estimated volatility-of-volatility from 0.011 to 0.032. Every SV number in the final results comes from the corrected model.

Without the simulation test, I would have reported results from a model that understated how much volatility moves by a factor of three, and nothing in the real-data output would have looked wrong.

## Giving the filter a second sensor

The failure analysis pointed to the fix. If returns are a weak signal about volatility, add a stronger one. The VIX is a second, independent reading of the same hidden state, but a biased one: it includes a variance risk premium. I built a bivariate state-space model in which returns and the VIX both measure log-variance, and the premium is its own persistent hidden state. The estimated premium persistence was 0.997. If I had modeled the premium as white noise, the filter would have simply copied the VIX.

This model cut the pure SV model's QLIKE loss by 19% (significant, p = 0.005), had the lowest mean absolute error of all eight models, and was the best model over 2021–2026. On QLIKE and squared error it was statistically tied with GJR and HAR + VIX, so it does not beat them; it joins them.

I have to be honest about the timing: I designed this model *after* seeing the first one fail. Its out-of-sample period is therefore not a clean test of the idea in the way the original hypothesis was, and the README says so. The fair claim is that fusing implied and realized information inside one filter is competitive with the best simple models and is worth testing properly on new data.

## What look-ahead is worth

The most important lesson for me personally came from a deliberate mistake. I evaluated the same SV model three ways. Honest walk-forward: QLIKE 0.428. Full-sample parameters: 0.385. Full-sample parameters plus Kalman-smoothed states: 0.170, with the Mincer–Zarnowitz R² jumping from 0.39 to 0.71.

The smoothed version looks like a breakthrough, and it is entirely an artifact of letting the model see the future. It is the same mistake I had to debug at FPT, where forecasts were being compared against full-sample smoothed components. This time I wrote the defense into the code: a test that corrupts all data after a cutoff and requires every model's forecasts before that date to stay identical to within 1e-10. The RTS smoother is included as a deliberate failing case, to prove the test can catch the problem.

## Accuracy is not edge

The variance risk premium was there: the VIX exceeded subsequent realized volatility on 83.6% of days. A short variance position struck at the VIX made money in most months and lost very large amounts in a few.

My pre-registered rule (sell volatility only when my forecast is at least 2 points below the VIX) raised the Sharpe ratio from 0.55 to 0.74 and reduced the maximum drawdown by about a third, but it earned less overall. It did not avoid the COVID loss, because no forecast saw COVID coming. The improvement came from sitting out months where implied volatility was cheap, not from avoiding crashes. The same rule driven by GJR-GARCH looked much better (Sharpe 1.21). I report it, but I did not pick it as my result, because choosing the best-looking signal after the fact is exactly how backtests become fiction. I also want to be clear that the trade itself is a simplification: the VIX is not exactly a variance-swap strike, and there are no real option prices, spreads, or hedging costs in this test.

## What I would do next

- **Better data.** Realized volatility from 5-minute returns instead of daily closes would reduce the noise in both the target and the SV observation equation.
- **A model that sees the sign.** SV with correlated return and volatility shocks captures the leverage effect, but it is no longer linear-Gaussian. That requires a particle filter or MCMC instead of a Kalman filter.
- **A jump component.** The failures were transitions, and a smooth AR(1) cannot jump.
- **Real option prices.** Fitting a full implied-volatility surface, and testing whether forecasts predict delta-hedged option returns rather than a synthetic swap, is where this becomes a real trading question.

Each of these requires mathematics I can use but cannot yet derive properly: Itô calculus behind continuous-time SV models like Heston, the link between variance swaps and the options strip, and filtering for nonlinear state-space models. That is a large part of why I want to do the MSFM.

## What I take away

I started this project believing that a more structural model would win. What I learned is more useful: the structure is only as good as the information flowing into it, the loss function encodes what you care about, and the most convincing result in a volatility backtest is usually the one that quietly used the future. The habits that made this work credible (stating the hypothesis first, testing the estimator on simulated data, guarding against look-ahead in code) are the same ones that turned a 68% forecast error into 14% at FPT. Here they kept me from fooling myself.
