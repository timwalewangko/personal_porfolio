import pandas as pd
import xgboost as xgb
from modules.utils import create_features
from xgboost import plot_importance
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


def generate_forecast(sales, region, train_pctg=0.75, periods=52):

    sales = pd.read_csv("data/sales.csv", index_col="week", parse_dates=True)
    sales = sales[region]

    # split % of data for train
    split_index = sales.index[int(len(sales) * train_pctg)]

    sales_train = pd.DataFrame(sales.loc[sales.index <= split_index].copy())
    sales_test = pd.DataFrame(sales.loc[sales.index > split_index].copy())

    X_train, y_train = create_features(sales_train, label=region)
    X_test, y_test = create_features(sales_test, label=region)

    xgb_reg = xgb.XGBRegressor(n_estimators=1000)
    xgb_reg.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    importance_plot = plot_importance(xgb_reg, height=0.9)
    sales_test["sales_prediction"] = xgb_reg.predict(X_test)

    _rmse = mean_squared_error(
        y_true=sales_test[region], y_pred=sales_test["sales_prediction"]
    )
    _mae = mean_absolute_error(
        y_true=sales_test[region], y_pred=sales_test["sales_prediction"]
    )
    _mape = mean_absolute_percentage_error(
        y_true=sales_test[region], y_pred=sales_test["sales_prediction"]
    )

    end_of_sales = sales.index[-1]
    new_start = end_of_sales + pd.Timedelta(days=7)  # type: ignore
    new_index = pd.date_range(start=new_start, periods=periods, freq="W")

    pred_df = pd.DataFrame(index=new_index, data=None)
    X_pred = create_features(pred_df)
    pred_df["forecast"] = xgb_reg.predict(X_pred)
    pred_df = pred_df.drop(columns=["date", "quarter", "month", "year", "weekofyear"])

    forecast = pd.concat([sales, pred_df])

    return forecast, importance_plot, [_rmse, _mae, _mape]
