"""Estimate used-car retail prices and compare two auction bidding rules."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SEED = 42
PREP_COST = 1_500
FULL = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]
V1 = ["Mileage_K", "Age_Years", "Horsepower", "Accident"]


def predictors(cars, features):
    x = cars.loc[:, features].copy()
    x["Accident"] = x["Accident"].map({"no": 0, "yes": 1})
    if x.isna().any().any():
        raise ValueError("Missing predictors or unrecognized accident history.")
    return x


def fit_ols(train, features):
    x = predictors(train, features)
    y = train["Sale_Price"]
    model = LinearRegression().fit(x, y)
    design = np.column_stack([np.ones(len(x)), x])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("OLS coefficient tests require a full-rank design.")
    residuals = y.to_numpy() - model.predict(x)
    df = len(y) - design.shape[1]
    mse = residuals @ residuals / df
    # Diagonal of (X'X)^-1 gives the variance factors for classical OLS t tests.
    variance_factors = np.diag(np.linalg.inv(design.T @ design))
    se = np.sqrt(mse * variance_factors)
    beta = np.r_[model.intercept_, model.coef_]
    coefficients = pd.DataFrame({
        "term": ["Intercept", *features],
        "estimate": beta, "std_error": se, "p_value": 2 * t.sf(np.abs(beta / se), df)
    }).set_index("term")
    r2 = model.score(x, y)
    adjusted_r2 = 1 - (1 - r2) * (len(y) - 1) / df
    return model, coefficients, adjusted_r2


def buy_or_pass(car_features, auction_price, price_model):
    """Use an already fitted price model to decide on one car at its auction price."""
    row = predictors(pd.DataFrame([car_features]), V1)
    predicted_retail = price_model.predict(row)[0]
    return "buy" if predicted_retail > auction_price + PREP_COST else "pass"


def test_fit_chart(predictions, path):
    actual = predictions["Sale_Price"]
    fitted = predictions["v1_prediction"]
    mean_price = predictions["baseline_prediction"].iloc[0]
    bounds = [actual.min(), actual.max(), fitted.min(), fitted.max(), mean_price]
    low = 5_000 * np.floor(min(bounds) / 5_000)
    high = 5_000 * np.ceil(max(bounds) / 5_000)
    fig, ax = plt.subplots(figsize=(6.8, 5.7), layout="constrained")
    ax.set_axisbelow(True)
    ax.grid(color="#e3e6e8", linewidth=0.7)
    ax.scatter(actual, fitted, color="#146574", alpha=0.80, s=30, edgecolor="white",
               linewidths=0.5, zorder=3, label="v1 prediction")
    ax.plot([low, high], [low, high], color="#58616a", linewidth=1.3,
            linestyle="--", label="Perfect prediction")
    ax.axhline(mean_price, color="#a74f20", linewidth=1.8, label="Training-mean baseline")
    ax.set(xlim=(low, high), ylim=(low, high), aspect="equal",
           xlabel="Actual retail sale price", ylabel="Predicted retail sale price")
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(MultipleLocator(10_000))
        axis.set_major_formatter(FuncFormatter(lambda val, _: f"${val/1000:g}k"))
    ax.set_title(f"Retail price predictions on {len(predictions)} held-out cars",
                 loc="left", pad=13, fontweight="semibold", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def auction_results(lots, models):
    # Keep inputs for bidding separate from the retail outcome used for scoring.
    features = lots[V1].to_dict("records")
    x = predictors(lots, V1)
    audit = lots.copy()
    for name, model in models.items():
        audit[f"{name}_prediction"] = model.predict(x)
        audit[f"{name}_decision"] = [
            buy_or_pass(car, price, model) for car, price in zip(features, lots["Auction_Price"])
        ]
    realized = lots["Later_Sale_Price"] - lots["Auction_Price"] - PREP_COST
    for name in models:
        bought = audit[f"{name}_decision"].eq("buy")
        audit[f"{name}_profit"] = realized.where(bought, 0)
    return audit


def money(amount, places=0):
    return f"{'-' if amount < 0 else ''}${abs(amount):,.{places}f}"


def main():
    OUT.mkdir(exist_ok=True)
    sales = pd.read_csv(ROOT / "used_car_sales.csv")
    train, test = train_test_split(sales, test_size=0.30, random_state=SEED)
    _, full_stats, full_adj = fit_ols(train, FULL)
    model, v1_stats, v1_adj = fit_ols(train, V1)
    baseline = DummyRegressor(strategy="mean").fit(predictors(train, V1), train["Sale_Price"])
    models = {"baseline": baseline, "v1": model}

    test_x = predictors(test, V1)
    predictions = test.copy()
    # Source row 1 is the first sale below the CSV header.
    predictions.insert(0, "sale_row", predictions.index + 1)
    scores = {}
    for name, policy in models.items():
        p = policy.predict(test_x)
        predictions[f"{name}_prediction"] = p
        scores[name] = (r2_score(test["Sale_Price"], p),
                        np.sqrt(mean_squared_error(test["Sale_Price"], p)))
    predictions.sort_values("sale_row").to_csv(OUT / "test_predictions.csv", index=False)
    test_fit_chart(predictions, OUT / "test_fit.png")

    lots = pd.read_csv(ROOT / "auction_lots.csv")
    auction = auction_results(lots, models)
    auction.to_csv(OUT / "auction_decisions.csv", index=False)
    counts, losses, profits = {}, {}, {}
    for name in models:
        bought = auction[f"{name}_decision"].eq("buy")
        profit = auction[f"{name}_profit"]
        counts[name] = int(bought.sum())
        losses[name] = int((bought & profit.lt(0)).sum())
        profits[name] = profit.sum()

    effects = {
        "Intercept": "Reference prediction with all numeric inputs zero and no accident",
        "Mileage_K": "Change for an additional 1,000 miles",
        "Age_Years": "Change for one additional year of age",
        "Horsepower": "Change for one additional horsepower",
        "Accident": "Difference for an accident on record compared with none",
        "Num_Owners": "Change for one additional previous owner",
    }
    lines = [
        "# Used-car pricing and auction results", "", "## Data and split", "",
        f"The course dataset contains {len(sales)} retail sales from the dealer's last year. "
        f"A random 70/30 split with seed {SEED} assigns {len(train)} cars to training and "
        f"{len(test)} to testing. The separate auction dataset contains {len(lots)} lots "
        "from one recent auction. Its later retail prices come from a market data service.", "",
        "## Training fit", "",
        "Both models use ordinary least squares with an intercept and retail sale price in dollars "
        "as the outcome. Accident is 1 for yes and 0 for no. Each coefficient holds the other "
        "included features fixed. P-values use two-sided OLS t tests.", "",
        "| Term | Full estimate | Full p-value | v1 estimate | v1 p-value | Meaning |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for term in full_stats.index:
        full = full_stats.loc[term]
        chosen = "— | —"
        if term in v1_stats.index:
            stat = v1_stats.loc[term]
            chosen = f"{money(stat.estimate, 2)} | {stat.p_value:.3g}"
        lines.append(f"| {term} | {money(full.estimate, 2)} | {full.p_value:.3g} | "
                     f"{chosen} | {effects[term]} |")
    lines += [
        "", f"Training adjusted R² is {full_adj:.5f} with all five inputs and "
        f"{v1_adj:.5f} after removing Num_Owners. The owner-count estimate is "
        f"{money(full_stats.loc['Num_Owners', 'estimate'], 2)} per additional owner "
        f"(p = {full_stats.loc['Num_Owners', 'p_value']:.3f}). "
        "The four-feature model is v1. It uses one fewer input and has slightly higher adjusted R². "
        "Each retained slope has p < 0.001. The intercept is an extrapolation because a car "
        "with zero horsepower is outside the observed range.", "",
        "## Held-out test fit", "",
        f"The baseline predicts the training mean of {money(train['Sale_Price'].mean(), 2)} "
        "for every car. Test R² uses the mean of the test outcomes as the reference for "
        "scoring both predictions. The training and test means differ, so the baseline "
        "can score slightly below zero.", "",
        "| Price estimate | Test R² | Test RMSE |", "|---|---:|---:|",
    ]
    for name, (r2, rmse) in scores.items():
        lines.append(f"| {name} | {r2:.5f} | {money(rmse, 2)} |")
    lines += [
        "", "![Predicted and actual retail prices for the held-out cars](test_fit.png)", "",
        "The diagonal represents an exact prediction. Points above it indicate overestimates. "
        "The horizontal line is the baseline prediction for every car.", "",
        "## Auction backtest", "",
        f"Buy when predicted retail price exceeds the auction price plus {money(PREP_COST)} "
        "in preparation costs. Pass at equality. For a purchase, realized profit equals "
        "later retail price minus the auction price and preparation cost. A pass earns zero. "
        "A purchase counts as a loss only when its realized profit is negative.", "",
        "| Buying rule | Cars bought | Loss-making purchases | Total realized profit |",
        "|---|---:|---:|---:|",
    ]
    for name in models:
        lines.append(f"| {name} | {counts[name]} | {losses[name]} | {money(profits[name])} |")
    lines += ["", f"Incremental profit over the baseline for these {len(lots)} lots: "
              f"{money(profits['v1'] - profits['baseline'])}.", ""]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Results written to {OUT}")


if __name__ == "__main__":
    main()
