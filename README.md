Japan Export Value Prediction

1. Project Overview
    This project predicts the monetary Value of an exported commodity from Japan. The prediction is based on the Year of export, the destination Country, and the Quantity of the item.
    The analysis involves:
    1. Loading and merging export data from 2021, 2022, 2023, and 2024.
    2. Cleaning and processing HS (Harmonized System) commodity codes.
    3. Merging export data with HS code descriptions to create a master dataset.
    4. Visualizing the data to find trends.
    5. Isolating a specific commodity ("General machinery") for focused modeling.
    6. Applying scaling and transformations (RobustScaler, Log, Yeo-Johnson) to normalize the features.
    7. Training and evaluating four different regression models to predict the export Value.

2. Dataset
    This project uses two main data sources:

    Japan Export Data (2021-2024): Four separate CSV files, one for each year. These files contain the Year, HS Code, Country, Units, Quantity, and Value for each export transaction.
    HS Code List: A CSV file (HS_Code_List.csv) used to map the 6-digit HS Code to its corresponding item description (Overview of items).

3. Libraries & Dependencies
    This project uses the following Python libraries. You can install them using pip:

    Bash
    pip install numpy pandas scikit-learn matplotlib seaborn lightgbm xgboost
    
    numpy & pandas: For data manipulation and analysis.
    re: For regular expression operations during data cleaning.
    matplotlib & seaborn: For data visualization.
    scipy.stats: For the Yeo-Johnson transformation.
    scikit-learn: For:
        RobustScaler (preprocessing)
        train_test_split (model training)
        LinearRegression (modeling)
        RandomForestRegressor (modeling)
        Metrics (mean_squared_error, r2_score, etc.)
        xgboost: For the XGBoost Regressor model.
        lightgbm: For the LightGBM Regressor model.

4. Methodology
    Data Preprocessing
        Concatenation: The four yearly export CSVs (2021-2024) are loaded, duplicates are dropped from each, and they are concatenated into a single DataFrame.
        HS Code Cleaning: The 9-digit HS Code from the export data is split. The first 6 digits (international standard) are kept as the HS Code, and the last 3 digits (Japan's domestic code) are saved in a new Domestic code column.
        HS Code Mapping: The HS_Code_List.csv is loaded. Its HS Code column, which contains ranges (e.g., 0201~0208), is processed to expand these ranges into individual 6-digit codes. This cleaned list is then merged with the main export data to add the Overview of items description to each record.
        Final Cleaning: Rows with no Overview of items (due to missing mappings) are dropped. Rows with a Quantity of 0 are also removed.
    
    Exploratory Data Analysis (EDA)
        The notebook visualizes the data to find insights, including:
        Total export value by year (Bar Chart).
        Top 10 countries by export value (Pie Chart).
        Top 10 exported items by value and quantity (Bar Charts).
        Model Training: "General Machinery"
        Due to the large variance in the full dataset, the project focuses on a single item, "General machinery", for predictive modeling.
    
    Feature Scaling:
        RobustScaler is applied to all numeric features to handle outliers.
        Log Transformation (np.log(x+1)) is applied to the Value and Quantity columns.
        Yeo-Johnson Transformation is applied to the Year, Country, Quantity_log, and Value_log columns to normalize their distributions.
    
    Feature Selection: 
        Features (X): Year_YJ, Country_YJ, Quantity_YJ
        Target (y): Value_YJ
        Model Evaluation: The data is split into 80% training and 20% testing sets. Four models are trained and evaluated:

    Linear Regression
        Random Forest Regressor
        XGBoost Regressor
        LightGBM Regressor

5. Results
    The models are evaluated on their ability to predict the transformed Value. Key metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), RMSE, and R-squared (R²). The notebook also performs a check for overfitting by comparing training and testing R² scores.

    Based on the notebook's final evaluation (for Linear Regression):
        Training R²: 0.75
        Testing R²: 0.75

Conclusion: The Linear Regression model is generalizing well for this specific item.

(Evaluation results for Random Forest, XGBoost, and LightGBM are also calculated).