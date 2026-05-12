American Home Crisis: Can Affordability be Predicted?
By: Ryan Bradley, Sean Catarroja, Abhinav Chede, Zach Rose, & Luke Tu
rbradley4@wisc.edu, catarros@lafayette.edu, abhinav_chede@brown.edu, zach.l.rose@gmail.com, & luke_tu@brown.edu
Background
In recent years, home affordability has become one of young Americans’ biggest concerns as they enter adulthood. Rising property prices, fluctuating mortgage rates, and income not scaling proportionally have made it increasingly difficult for first-time buyers to enter the market. Because of this, people are asking the same question:

“Can I afford my monthly payment?”

National housing statistics are not representative of what a resident in Durham, North Carolina may be facing vs someone in San Francisco may be dealing with. Housing affordability is not the same everywhere; in fact, it can vary from county to county. Some counties are becoming much more unaffordable much faster than others, while others may be improving. This raises the question:

	“Can we predict how housing markets are changing in affordability county-by-county?"

This project aims to answer just that through machine learning and specifically neural networks to forecast the short-term housing affordability at the county level. Most existing housing forecasts focus solely on home prices, or give larger scale trends, such as at the state or nation level. This just isn’t specific enough to be helpful to the average consumer. Everyday Americans need to take three things, all localized to county, into consideration: the average percentage of income a resident spends on housing, mortgage rates, and median income. 


	We chose to have our forecasting models target two different metrics. First, we wanted to predict the actual affordability of a county. Then, we wanted to predict the percent change in affordability. We would compare which metric the models could more accurately predict. These two targets would also give us context as to how expensive a city actually is and then how quickly it is becoming more or less affordable.

Target Metric Selection
Choosing what to predict turned out to be one of the most important decisions of the whole project. We went through three iterations before landing on our final setup.

Attempt 1: Our Own Formula
Our first idea was to build our own affordability score from scratch using the raw data we already had. The formula was:
Affordability Score = (Monthly Income / Monthly Mortgage Payment) x 100
A higher score means a county is more affordable. A lower score means people are spending a bigger chunk of their income on a mortgage. We calculated the mortgage payment using the standard amortization formula, assuming 20% down on a 30-year loan at the current rate.
The problem we quickly ran into: our input features (income, home price, mortgage rate) are the exact same numbers that go into computing this score. That means a model trained to predict it would basically just learn the formula itself rather than discovering any real patterns in the data. It would be like asking a calculator to predict the answer to an equation it already knows.

Attempt 2: Switching to Zillow's Affordability Metric
To fix that problem, we switched our target variable to Zillow's own Affordability Metric. Zillow computes this score using their own methodology, which factors in local market conditions in a way that is not a direct algebraic function of our input features. This gave the model something genuinely harder to learn, so we could trust that any accuracy it achieved came from real pattern recognition and not just reverse-engineering a formula.

Attempt 3: Adding Percent Change
Even with Zillow's metric as our target, the raw score moves slowly and predictably from month to month for most counties. To make the prediction task harder and more useful, we also trained models to predict the percent change in the affordability score from one month to the next rather than the raw score itself.
Predicting percent change is fundamentally tougher because it requires the model to anticipate shifts and turning points, not just track the current level. But it is also more actionable: knowing a county is about to get 5% less affordable is more useful than just knowing its current number. This gave us four models in total: two predicting the raw Zillow Affordability Metric (one FFN, one LSTM) and two predicting next-month percent change (one FFN, one LSTM).

Data Selection & Pipeline
We combined and cleaned four datasets into one to test our models on.


   Zillow county home value
Zillow's estimate of typical home value per county over time, our primary measure of local home prices
https://www.zillow.com/research/data/
   FRED County income data
Average personal income per county, used to measure local earning power relative to housing costs
https://fred.stlouisfed.org/release?rid=175
   FRED 30-year mortgage rates
National average fixed rate for a 30-year mortgage, the key driver of monthly payment costs
https://fred.stlouisfed.org/series/MORTGAGE30US
   Zillow % income going to mortgage
share of income spent on housing per county per month
https://www.zillow.com/research/data/




We joined these four datasets using the Federal Information Processing Standards (FIPS) codes as a key. FIPS codes are standardized numerical identifiers of geographical regions of the U.S. They are 5 digit codes; the first two numbers identify the state and the following three numbers identify the country. With this joined dataset, we created a 3D dataset. From this we use the FIPS codes as one dimension,  time, in months, as the second dimension, and our additional features as the third dimension. Again the county and time-specific features we are evaluating are the percentage of income a resident spends on housing, mortgage rate, and median income.

On top of the raw joined data, we built additional features to help the models learn from patterns over time rather than just a single snapshot. For each variable we added a 1-month lag, a 3-month rolling average, and a 12-month rolling average. We also included sine and cosine encodings of the month to capture any seasonal patterns. FIPS codes were handled as learned embeddings so the models could pick up on county-level differences without treating the code as a plain number.
One important limitation here is that income from FRED is reported annually at the county level, not monthly. This means the income feature only steps up once per year while everything else updates every month. This creates an imbalance in signal quality that we worked around using the rolling averages, but it is worth keeping in mind as a constraint on the percent change models in particular.

Models & Results:
We decided to train four models. Two Feedforward Neural Networks (FFN) and two Long Short-Term Memory networks. Between the two versions of each model they will have two prediction targets; one predicting affordability level and the other predicting the % change in affordability
We chose to use a feedforward neural network as a comparison neural model for the LSTM because it can learn relationships between key variables like home prices, income, and mortgage rates without explicitly modeling time. This gives us a useful comparison to more complex time-series models like LSTMs.
We chose to use LSTMs since our project’s focus on housing affordability changes over time, where past patterns and trends affect those changes. LSTM’s are designed to be able to recognize ese sequential trends and adjust its weights based on them. We believe that this makes them more well suited to predict the time-series data we’ve curated on monthly housing data. 


Baselines
For each model we defined a naive baseline to compare against. For the FFN predicting the raw affordability score, the baseline was the training set mean, meaning it just predicted the same constant value for every county every month. For the LSTM predicting the raw score, the baseline was the previous month's actual score, which assumes next month will equal this month. For both FFN and LSTM on the percent change task, the baseline was the current month's percent change, meaning it assumed next month's change would equal this month's change. These baselines represent the simplest possible predictions, so beating them is the minimum bar for a model to be considered useful.

FFN:
The FFN predicting the raw Zillow Affordability Score achieved an R² of 0.51, with a baseline R² of -0.28. The loss drops sharply in the first epoch and stabilizes quickly, showing the model learned the main structure of the data. The scatter plot shows predictions tracking the general trend with some spread, well ahead of simply predicting a constant value.


Figure 1: FFN model predicting the Zillow Affordability Score. R² = 0.5132, Baseline R² = -0.2764. Loss drops sharply in the first epoch and stabilizes. The FNN predictions (blue) track the diagonal much better than the flat baseline (orange).

The FFN predicting next-month percent change achieved an R² of 0.32. Both train and test loss decrease steadily over 14 epochs, a sign the model is genuinely learning rather than overfitting. It captures the direction of change for most counties but undershoots the largest spikes, which are inherently hard to predict.


Figure 3: FFN model predicting next-month affordability percent change. R² = 0.3166. Both train and test loss decrease steadily over 14 epochs. The model captures the general direction of change for most counties, though extreme spikes are undershot.



LSTM:

The LSTM predicting the raw Zillow Affordability Score achieved an R² of 0.86, making it the strongest model we built. The baseline R² of -0.29 confirms how far ahead the LSTM is of simply predicting a constant. The scatter plot shows predictions clustered tightly around the perfect prediction line across most of the range, with the residual plot showing errors centered near zero for the large majority of counties.


Figure 2: LSTM model predicting the Zillow Affordability Score. R² = 0.8558, Baseline R² = -0.2860. Loss decreases steadily over 25 epochs. Predictions cluster tightly around the perfect prediction line, far outperforming the constant baseline.

The county examples show that for highly volatile counties, the LSTM often tracks the central trend but misses sharp sudden spikes. The baseline is also plotted, and the LSTM consistently improves on it in normal periods, though both struggle equally on extreme one-month events.
The LSTM predicting next-month percent change achieved an R² of 0.43, the strongest percent change result of any model. The scatter plot shows the Direct LSTM (orange) clustering more tightly around the perfect prediction line than the naive baseline (blue). The 12-month sequential lookback gives the LSTM a real advantage for this task compared to the FFN.


Figure 4: LSTM model predicting next-month affordability percent change. R² = 0.4291. The Direct LSTM (orange) clusters more tightly around the perfect prediction line than the current-month-change baseline (blue), especially in the typical range of values.



Figure 5: LSTM percent change predictions for three counties (FIPS 35057, 42021, 17005) from 2021 to 2025. The Direct LSTM (green) tracks gradual trends well and outperforms the naive baseline (orange) in most periods. Sudden extreme spikes remain difficult to predict for any model.


Findings

Model                           Task                                    R²                Verdict
LSTM (Affordability Metric)     Raw affordability prediction          0.8558               Best
FFN (Affordability Metric)      Raw affordability prediction          0.5132               Good
FFN (Percent Change)            Next-month % change                   0.3166              Moderate
LSTM (Percent Change)           Next-month % change                   0.4291              Strong





It is evident that predicting the raw affordability metric is considerably easier and achieves strong results regardless of the model. Predicting next-month percent change is inherently harder as both models struggle badly, especially the LSTM. From our results we can see that the task had a much bigger impact on the forecasting than the choice of model.

Between the two predicting the actual metric, the LSTM unsurprisingly outperformed the FFN, but only marginally. The LSTM's temporal modeling definitely helps it capture trends better, but
the FFN still performs well enough to be useful.


On the percent change task, the LSTM outperformed the FFN (R2 of 0.43 vs 0.32). This makes sense: predicting how affordability will shift next month benefits from looking at the trajectory of recent months, which is exactly what the LSTM's 12-month lookback window is designed to capture. The FFN, which only sees a single snapshot of features, cannot pick up on those directional trends as effectively.

Percent change is a fundamentally harder problem than predicting the raw score. The best percent change model (LSTM at R2 = 0.43) still only explains 43% of the variance, compared to 51-86% for the raw score models. The percent change distribution has heavy tails with large spikes that neither model predicts well. Both models tend to regress toward the mean and miss the extreme events that are often the most important ones for real-world decision making.

Key Takeaways
If you need a short-term affordability point estimate, use the LSTM on the raw Zillow Affordability Metric (R2 = 0.86). It is the most reliable model we built. If you need to predict next-month directional change, the LSTM percent change model is the better choice (R2 = 0.43), and it is meaningfully more useful than the FFN for this task. Across both tasks, the LSTM outperformed the FFN, which suggests that the sequential nature of housing data genuinely benefits from a model that can learn from time patterns.

Improving the percent change models would likely require better feature engineering, such as adding policy change signals, regional housing inventory data, or local unemployment rates. It may also benefit from target preprocessing like log-transforming or clipping the percent change values to reduce the influence of extreme outliers during training.

Discussion

Our results tell a clear story about what is easy and what is hard when predicting housing affordability.
Predicting the raw Zillow Affordability Metric worked well because housing markets move slowly most of the time. A county's affordability score in any given month is usually close to what it was the previous month, so models that learn from recent history can do well. The LSTM achieved the strongest result overall with an R2 of 0.86, while the FFN reached 0.51. Both were well ahead of their respective naive baselines (both at around -0.28), which confirms the models are learning real structure in the data. Importantly, these results held up after we switched from our custom formula to Zillow's metric as the target, meaning the models learned real relationships rather than just recomputing a formula from its inputs.
Predicting percent change was harder than predicting the raw score, as expected. Month-to-month swings in affordability are noisy and sensitive to sudden external shocks like a rapid shift in mortgage rates. That said, the results were better than our earlier runs. We experimented with different lookback windows, including 3 months, before settling on 12 months as it produced the best validation performance. The LSTM achieved an R2 of 0.43 on this task, a meaningful result that shows the 12-month sequential lookback genuinely helps when predicting directional change. The FFN reached 0.32. Both models beat the naive baseline of repeating this month's change. The LSTM's advantage on percent change confirms that knowing the recent trajectory of affordability, not just the current snapshot, is important for predicting how it will move next.
The decision to test both the raw score and percent change as targets was valuable. If we had only predicted the raw score, we would have had strong numbers but less useful outputs. If we had only predicted percent change, we would have had weaker numbers but more actionable outputs. Having both gives a more complete picture of what machine learning can and cannot do for housing affordability forecasting at the county level.
The annual income data from FRED remains a meaningful limitation. Because income only updates once a year while everything else is monthly, the income feature has lower temporal resolution than the rest. This almost certainly limits the percent change models more than the score models, since change prediction depends more heavily on detecting short-term signals.





Conclusion
Buying a home is one of the biggest financial decisions most people will ever make, and it is getting harder every year in many parts of the country. Our project used machine learning to try to give people and policymakers a better way to see where affordability is heading before it gets there.
We found that predicting the Zillow Affordability Metric one month ahead is very achievable with neural networks, with R2 values of 0.51 and 0.86 for the FFN and LSTM respectively. Predicting percent change in affordability is harder but produced strong results too, with the LSTM reaching an R2 of 0.43 and the FFN reaching 0.32. Across both tasks the LSTM outperformed the FFN, confirming that the sequential nature of monthly housing data is something a time-aware model can genuinely exploit. The choice to move away from our own formula as the target and use Zillow's metric instead made the results more trustworthy, and testing percent change as a second target made them more practically useful.
The biggest lesson from this project is that housing affordability is highly local and highly time-dependent. A single national statistic hides enormous variation county by county. Models that work at the county level using time-series data give a much clearer picture of what is actually happening on the ground. At the same time, the hardest affordability events to predict, sudden large spikes, are exactly the ones that matter most for real people making housing decisions. Closing that gap is the most important direction for future work.
All of our code, data processing steps, and figures are available at: https://github.com/zrose3-zr/Affordable-housing-final


