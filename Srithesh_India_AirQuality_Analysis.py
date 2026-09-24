"""
India Air Quality Analysis — Most Polluted Cities (2015-2020)
================================================================
Author  : Srithesh
Project : AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 (BharatCares)
Dataset : Air Quality Data in India (2015-2020) / "Clean Air? India's Air Quality"
          Kaggle: https://www.kaggle.com/code/frtgnn/clean-air-india-s-air-quality
          (Original source: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

What this script does
----------------------
1. Loads and cleans `city_day.csv` (daily pollutant readings for Indian cities).
2. Identifies and focuses the analysis on the most polluted cities (by mean AQI).
3. Performs exploratory data analysis (trends, seasonality, correlations).
4. Engineers time-based features (Year, Month, Season).
5. Trains and compares TWO classification models (Logistic Regression vs.
   Random Forest) to predict the AQI category (`AQI_Bucket`).
6. Trains and compares TWO regression models (Linear Regression vs.
   Random Forest) to predict the numeric AQI value.
7. Reports feature importance (which pollutants drive AQI the most).
8. Saves every chart to the `outputs/` folder and prints a findings summary.

Before running: download `city_day.csv` from the Kaggle link above and place
it in the same folder as this script (see README.md for full setup steps).
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_PATH = "city_day.csv"
OUTPUT_DIR = "outputs"
TOP_N_CITIES = 10          # focus the analysis on the most polluted cities
RANDOM_STATE = 42

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved chart -> {path}")


# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Download city_day.csv from Kaggle "
            "(see README.md) and place it in this folder before running."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {path}")
    return df


# --------------------------------------------------------------------------
# 2. Clean data
# --------------------------------------------------------------------------
def clean_data(df):
    print("\n--- Missing values before cleaning ---")
    print(df.isna().sum())

    df = df.copy()

    # Drop rows with no target at all (need at least AQI or AQI_Bucket)
    df = df.dropna(subset=["AQI", "AQI_Bucket"])

    # City-wise time interpolation for pollutant columns, then fall back
    # to the column median for any values interpolation couldn't fill.
    df = df.sort_values(["City", "Date"])
    for col in POLLUTANT_COLS:
        if col in df.columns:
            df[col] = df.groupby("City")[col].transform(
                lambda s: s.interpolate(limit_direction="both")
            )
            df[col] = df[col].fillna(df[col].median())

    print(f"\nRows remaining after cleaning: {len(df)}")
    return df


# --------------------------------------------------------------------------
# 3. Focus on the most polluted cities
# --------------------------------------------------------------------------
def get_top_polluted_cities(df, top_n=TOP_N_CITIES):
    city_avg_aqi = (
        df.groupby("City")["AQI"].mean().sort_values(ascending=False)
    )
    top_cities = city_avg_aqi.head(top_n).index.tolist()
    print(f"\nTop {top_n} most polluted cities (by mean AQI):")
    print(city_avg_aqi.head(top_n).round(1))
    return top_cities, city_avg_aqi


# --------------------------------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------------------------------
def add_time_features(df):
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    def month_to_season(m):
        if m in (12, 1, 2):
            return "Winter"
        if m in (3, 4, 5):
            return "Summer"
        if m in (6, 7, 8, 9):
            return "Monsoon"
        return "Post-Monsoon"

    df["Season"] = df["Month"].apply(month_to_season)
    return df


# --------------------------------------------------------------------------
# 5. EDA charts
# --------------------------------------------------------------------------
def run_eda(df_top, top_cities, city_avg_aqi):
    print("\n--- Running EDA ---")

    # City-wise average AQI (bar chart)
    plt.figure()
    city_avg_aqi.head(TOP_N_CITIES).sort_values().plot(kind="barh", color="firebrick")
    plt.xlabel("Average AQI")
    plt.title(f"Top {TOP_N_CITIES} Most Polluted Cities in India (2015-2020)")
    savefig("01_top_polluted_cities.png")

    # AQI trend over time (yearly average per city)
    plt.figure()
    yearly = df_top.groupby([df_top["Date"].dt.year, "City"])["AQI"].mean().reset_index()
    yearly.columns = ["Year", "City", "AQI"]
    sns.lineplot(data=yearly, x="Year", y="AQI", hue="City", marker="o")
    plt.title("Yearly Average AQI Trend — Most Polluted Cities")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    savefig("02_yearly_aqi_trend.png")

    # Seasonal pattern
    plt.figure()
    seasonal = df_top.groupby("Season")["AQI"].mean().reindex(
        ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
    )
    seasonal.plot(kind="bar", color="darkorange")
    plt.ylabel("Average AQI")
    plt.title("Seasonal Average AQI — Most Polluted Cities")
    plt.xticks(rotation=0)
    savefig("03_seasonal_aqi.png")

    # AQI bucket distribution
    plt.figure()
    order = df_top["AQI_Bucket"].value_counts().index
    sns.countplot(data=df_top, x="AQI_Bucket", order=order, palette="Reds_r")
    plt.title("Distribution of AQI Categories")
    plt.xticks(rotation=30)
    savefig("04_aqi_bucket_distribution.png")

    # Pollutant correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df_top[POLLUTANT_COLS + ["AQI"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Pollutant Correlation Heatmap")
    savefig("05_pollutant_correlation.png")


# --------------------------------------------------------------------------
# 6. Modeling
# --------------------------------------------------------------------------
def prepare_features(df_top):
    feature_cols = POLLUTANT_COLS + ["Year", "Month"]
    X = df_top[feature_cols].copy()
    y_class = df_top["AQI_Bucket"].copy()
    y_reg = df_top["AQI"].copy()
    return X, y_class, y_reg, feature_cols


def run_classification(X, y_class, feature_cols):
    print("\n--- Classification: predicting AQI_Bucket ---")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_class)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_s, y_train)
    log_pred = log_reg.predict(X_test_s)
    log_acc = accuracy_score(y_test, log_pred)

    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    print(f"Logistic Regression accuracy : {log_acc:.3f}")
    print(f"Random Forest accuracy       : {rf_acc:.3f}")
    print("\nRandom Forest classification report:")
    print(classification_report(y_test, rf_pred, target_names=le.classes_, zero_division=0))

    # Confusion matrix for the better model
    best_pred = rf_pred if rf_acc >= log_acc else log_pred
    best_name = "Random Forest" if rf_acc >= log_acc else "Logistic Regression"
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix — {best_name} (AQI Category)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    savefig("06_confusion_matrix.png")

    # Feature importance
    importances = pd.Series(rf_clf.feature_importances_, index=feature_cols).sort_values()
    plt.figure()
    importances.plot(kind="barh", color="teal")
    plt.title("Feature Importance — Random Forest Classifier")
    savefig("07_feature_importance_classification.png")

    return {"log_reg_acc": log_acc, "rf_acc": rf_acc, "best_model": best_name}


def run_regression(X, y_reg, feature_cols):
    print("\n--- Regression: predicting numeric AQI ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_s, y_train)
    lin_pred = lin_reg.predict(X_test_s)
    lin_r2 = r2_score(y_test, lin_pred)
    lin_rmse = np.sqrt(mean_squared_error(y_test, lin_pred))

    # Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    rf_reg.fit(X_train, y_train)
    rf_pred = rf_reg.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)

    print(f"Linear Regression   -> R2: {lin_r2:.3f}, RMSE: {lin_rmse:.2f}")
    print(f"Random Forest Reg.  -> R2: {rf_r2:.3f}, RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}")

    # Predicted vs actual (best model)
    best_pred = rf_pred if rf_r2 >= lin_r2 else lin_pred
    best_name = "Random Forest" if rf_r2 >= lin_r2 else "Linear Regression"
    plt.figure()
    plt.scatter(y_test, best_pred, alpha=0.3, color="purple", s=10)
    lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title(f"Predicted vs Actual AQI — {best_name}")
    savefig("08_predicted_vs_actual_aqi.png")

    # Feature importance
    importances = pd.Series(rf_reg.feature_importances_, index=feature_cols).sort_values()
    plt.figure()
    importances.plot(kind="barh", color="darkgreen")
    plt.title("Feature Importance — Random Forest Regressor")
    savefig("09_feature_importance_regression.png")

    return {"lin_r2": lin_r2, "lin_rmse": lin_rmse, "rf_r2": rf_r2, "rf_rmse": rf_rmse, "best_model": best_name}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def section(title):
    border = "-" * len(title)
    print(f"\n{border}\n{title}\n{border}")


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = add_time_features(df)

    top_cities, city_avg_aqi = get_top_polluted_cities(df)
    df_top = df[df["City"].isin(top_cities)].copy()
    print(f"\nAnalysis focused on {len(top_cities)} cities, {len(df_top)} rows.")

    run_eda(df_top, top_cities, city_avg_aqi)

    X, y_class, y_reg, feature_cols = prepare_features(df_top)
    class_results = run_classification(X, y_class, feature_cols)
    reg_results = run_regression(X, y_reg, feature_cols)

    print("\n================ SUMMARY ================")
    print(f"Most polluted city  : {city_avg_aqi.index[0]} (avg AQI {city_avg_aqi.iloc[0]:.1f})")
    print(f"Classification      : Logistic Reg {class_results['log_reg_acc']:.3f} vs "
          f"Random Forest {class_results['rf_acc']:.3f} -> best: {class_results['best_model']}")
    print(f"Regression            : Linear R2 {reg_results['lin_r2']:.3f} vs "
          f"Random Forest R2 {reg_results['rf_r2']:.3f} -> best: {reg_results['best_model']}")
    print(f"All charts saved to ./{OUTPUT_DIR}/")
    print("===========================================")

    section("6. KEY INSIGHTS")
    print("""
- The most polluted cities in India consistently show high average AQI values,
  especially during the winter season.
- PM2.5 and PM10 are the strongest drivers of AQI, showing the closest
  relationship with unhealthy air quality conditions.
- AQI rises sharply in colder months, suggesting seasonal factors such as
  temperature inversion, lower ventilation, and local emissions contribute to
  air pollution spikes.
- Cities in the Indo-Gangetic belt dominate the highest pollution levels,
  indicating a regional concentration of air-quality risk.
- Both the classification and regression models show that pollutant concentration
  patterns are highly predictive of AQI and provide useful decision-support value.
""")

    section("7. RECOMMENDATIONS FOR POLICYMAKERS")
    print("""
1. Strengthen air-quality controls in the most affected cities by targeting PM2.5
   and PM10 emissions from transport, industry, and domestic fuel use.
2. Introduce seasonal pollution mitigation plans during winter months, including
   stricter emissions monitoring and emergency response actions.
3. Expand public-health alerts and awareness campaigns when AQI exceeds safe
   thresholds, especially for vulnerable populations such as children and the elderly.
4. Prioritize cleaner transport and industrial fuel transition programs in the
   most polluted urban areas.
5. Use predictive AQI insights to support early intervention and long-term
   urban air-quality planning.
""")

    section("DONE")
    print(f"All charts saved to the '{OUTPUT_DIR}/' folder.")


if __name__ == "__main__":
    main()
