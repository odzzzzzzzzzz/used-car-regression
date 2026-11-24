# Used-car prices and auction bids

This analysis estimates a dealer's retail sale prices and applies its buy-or-pass rule at a separate wholesale auction.

[Analysis results](results/results.md) · [Test-car predictions](results/test_predictions.csv) · [Decisions and profit for all 40 lots](results/auction_decisions.csv)

## Reproduce

Tested with Python 3.12. From this directory, run

```sh
python -m pip install -r requirements.txt
python analysis.py
```

The script uses the two original course CSV files in this folder. Running it rebuilds the results tables and the test-set scatterplot in `results/test_fit.png`.

## Files

- `used_car_sales.csv` contains historical retail transactions. A fixed random split assigns each transaction to training or testing. In the test predictions table, `sale_row` identifies each transaction by its one-based data-row number in this CSV.
- `auction_lots.csv` contains cars from one auction. Each decision uses the car's features, its observed auction price, and the fixed preparation cost. The later sale price enters only when scoring realized profit.
- `analysis.py` fits the baseline and both regression specifications, evaluates predictions, and runs the auction comparison. Its `buy_or_pass(car_features, auction_price, price_model)` function returns the decision for one bid.
