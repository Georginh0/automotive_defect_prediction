import matplotlib.pyplot as plt
import seaborn as sns
from .config import FIGURES_DIR


def plot_residuals(y_true, y_pred, save=True):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color="r", linestyle="--")
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, "residuals.png"), dpi=300)
    plt.show()


def plot_roc_curve(y_true, y_prob, save=True):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_true, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, "roc.png"), dpi=300)
    plt.show()


# Add more: e.g., feature importance barplot
