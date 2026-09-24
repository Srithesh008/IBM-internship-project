# India Air Quality — Most Polluted Cities Analysis

Exploratory data analysis and machine learning project that analyzes air quality data across Indian cities (2015–2020) to identify the most polluted cities and predict AQI, built for the **AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026**, conducted by BharatCares.

## Project Description

Air pollution is one of India's most pressing public health challenges. This project:

1. Cleans and processes daily pollutant readings (PM2.5, PM10, NO2, SO2, CO, O3, etc.) for cities across India.
2. Identifies the most polluted cities by average AQI and focuses the analysis on them.
3. Performs exploratory data analysis (EDA) — yearly trends, seasonal patterns, AQI category distribution, and pollutant correlations.
4. Trains and compares **two classification models** (Logistic Regression vs. Random Forest) to predict the AQI category (`AQI_Bucket`: Good → Severe).
5. Trains and compares **two regression models** (Linear Regression vs. Random Forest) to predict the numeric AQI value.
6. Surfaces the most important pollutants driving AQI, and provides public-health-focused recommendations.

## Dataset

**Air Quality Data in India (2015–2020)** — "Clean Air? India's Air Quality"
Source: [Kaggle — frtgnn/clean-air-india-s-air-quality](https://www.kaggle.com/code/frtgnn/clean-air-india-s-air-quality/input) (original dataset by [rohanrao/air-quality-data-in-india](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india))

- Daily pollutant readings (`city_day.csv`) from monitoring stations across Indian cities
- Columns: `City`, `Date`, `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`, `AQI`, `AQI_Bucket`
- Target variables: `AQI` (numeric) and `AQI_Bucket` (categorical: Good / Satisfactory / Moderate / Poor / Very Poor / Severe)

> Download `city_day.csv` from the Kaggle link above and place it in the same folder as the script before running.

## Technologies Used

- **Python 3**
- **pandas / numpy** — data cleaning and manipulation
- **matplotlib / seaborn** — data visualization
- **scikit-learn** — machine learning (Logistic Regression, Linear Regression, Random Forest, model evaluation)

## Project Structure

```
├── Srithesh_India_AirQuality_Analysis.py   # Main analysis script
├── requirements.txt                        # Python dependencies
├── Srithesh_ProjectReport.docx             # Full project report
├── README.md                               # This file
├── city_day.csv                            # Dataset (download separately from Kaggle)
└── outputs/                                # Charts saved here when the script runs
```

## Setup & Run Instructions

1. Clone this repository:

```
git clone <your-repo-url>
```

2. Create a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Download the dataset from [Kaggle](https://www.kaggle.com/code/frtgnn/clean-air-india-s-air-quality/input) and place `city_day.csv` in the project folder.
5. Run the script:

```
python Srithesh_India_AirQuality_Analysis.py
```

Progress and results print to the console, and all charts are saved as PNG files in the `outputs/` folder for later reference.

## Key Findings

> Generated from a schema-accurate test run used to validate the pipeline. Re-run the script on the full downloaded dataset to refresh these figures before final submission — see the note in the project report.

- North Indian cities (e.g. Delhi and other Indo-Gangetic Plain cities) consistently show the highest average AQI, driven mainly by PM2.5 and PM10.
- AQI spikes sharply in **winter months** (Nov–Feb) across the most polluted cities, consistent with stubble burning, temperature inversion, and lower wind dispersion.
- PM2.5 and PM10 are the strongest predictors of AQI in both the classification and regression models.
- The Random Forest and linear/logistic baselines perform comparably on this feature set, suggesting AQI here is largely explained by a near-linear combination of pollutant concentrations, with Random Forest adding robustness on minority AQI categories (e.g. "Severe").

## Author

*Srithesh, <sritheshp@gmail.com>*

## License

This project is for academic/educational purposes as part of the AICTE–IBM SkillsBuild Internship Program.
