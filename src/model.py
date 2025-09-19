import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
)
from imblearn.over_sampling import SMOTE
from scipy import stats
import statsmodels.api as sm
from .config import RANDOM_STATE, TEST_SIZE, LOG_C, TARGET_AUC


class DefectPredictor:
    def __init__(self):
        self.log_model = None
        self.multi_model = None
        self.simple_coefs = None

    def simple_regression(self, lot_df):
        """Identify high-leverage factors."""
        high_leverage = []
        for col in lot_df.columns[1:]:
            X = lot_df[[col]]
            y = lot_df["defect_count"]
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            if r2 > 0.5:
                high_leverage.append((col, r2, model.coef_[0]))
        self.simple_coefs = sorted(high_leverage, key=lambda x: x[1], reverse=True)
        return self.simple_coefs

    def multiple_regression(self, lot_df):
        """Multiple linear for defect counts."""
        train, test = train_test_split(lot_df, test_size=TEST_SIZE, shuffle=False)
        X_train = sm.add_constant(train.drop("defect_count", axis=1))
        y_train = train["defect_count"]
        self.multi_model = sm.OLS(y_train, X_train).fit()

        X_test = sm.add_constant(test.drop("defect_count", axis=1))
        y_test = test["defect_count"]
        y_pred = self.multi_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        return rmse, mae, self.multi_model.summary()

    def logistic_regression(self, df):
        """Logistic for unit failure."""
        train_idx = int(len(df) * (1 - TEST_SIZE))
        train_df, test_df = df.iloc[:train_idx], df.iloc[train_idx:]
        X_train = train_df.drop(["label", "timestamp", "lot_id"], axis=1)
        y_train = train_df["label"]
        X_test = test_df.drop(["label", "timestamp", "lot_id"], axis=1)
        y_test = test_df["label"]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        self.log_model = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=1000
        )
        grid = GridSearchCV(self.log_model, {"C": LOG_C}, cv=5, scoring="roc_auc")
        grid.fit(X_train_res, y_train_res)
        self.log_model = grid.best_estimator_

        y_prob = self.log_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)

        cv_scores = cross_val_score(
            self.log_model, X_train_res, y_train_res, cv=5, scoring="roc_auc"
        )
        return roc_auc, pr_auc, cv_scores.mean(), cv_scores.std()

    def save_models(self):
        joblib.dump(self, "models/automotive_predictor.pkl")
