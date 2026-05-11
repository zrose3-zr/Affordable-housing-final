# Affordable-housing-final

## Blog Figures

All figures used in the blog post are saved in the `blog_figures` folder. They are generated in the model notebooks with `fig.savefig(...)` after the plotting code.

Source locations: 

- `ffn_affordability_score_diagnostics.png`: generated in `notebooks/ffn_affordability_score.ipynb` by the final model-diagnostics plot, which shows training/validation loss and actual vs. predicted affordability score.
- `lstm_affordability_score_diagnostics.png`: generated in `notebooks/lstm_affordability_score.ipynb` by the final model-diagnostics plot, which shows training/validation loss and actual vs. predicted affordability score.
- `ffn_percent_change_diagnostics.png`: generated in `notebooks/ffn_affordability_percent_change.ipynb` by the main test-set diagnostics plot for the feed-forward percent-change model.
- `ffn_percent_change_residual_diagnostics.png`: generated in `notebooks/ffn_affordability_percent_change.ipynb` by the residual-analysis plot for the feed-forward percent-change model.
- `lstm_percent_change_diagnostics.png`: generated in `notebooks/lstm_affordability_percent_change_residual_tuned.ipynb` by the main test-set diagnostics plot comparing the direct LSTM model with the baseline.
- `lstm_percent_change_county_examples.png`: generated in `notebooks/lstm_affordability_percent_change_residual_tuned.ipynb` by the county-level time-series comparison plot.
- `lstm_percent_change_residual_diagnostics.png`: generated in `notebooks/lstm_affordability_percent_change_residual_tuned.ipynb` by the residual and prediction-distribution diagnostics plot.
