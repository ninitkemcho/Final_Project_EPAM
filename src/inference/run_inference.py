import os
import pickle
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Loading vectorized X_test and y_test
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','data', 'processed'))
X_test_tfidf = load_npz(os.path.join(base_dir, "X_test_tfidf.npz"))

fp_y_test = os.path.join(base_dir, 'y_test.csv')
y_test = pd.read_csv(fp_y_test)

# Loading model
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", 'models'))
model_path = os.path.join(output_dir, "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Results functions
def save_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Greens')

    fig_path = os.path.join(save_path, "confusion_matrix.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {fig_path}")
    
def save_metrics(y_true, y_pred, y_prob, save_path):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    auc_roc = roc_auc_score(y_true, y_prob)

    # Save metrics to a text file
    metrics_path = os.path.join(save_path, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy  : {accuracy:.4f}\n")
        f.write(f"Precision : {precision:.4f}\n")
        f.write(f"Recall    : {recall:.4f}\n")
        f.write(f"F1-score  : {f1:.4f}\n")
        f.write(f"AUC-ROC   : {auc_roc:.4f}\n")
        f.write("=" * 50 + "\n")

    print(f"Metrics saved to {metrics_path}")
    
# Predicting and saving results in outputs
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", 'predictions'))
os.makedirs(output_dir, exist_ok=True)

pred_path = os.path.join(output_dir, "predictions.csv")

y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)

pd.DataFrame({"y_pred": y_pred, "y_prob": y_prob[:, 1]}).to_csv(pred_path, index=False)
print(f"Predictions saved to {pred_path}")

figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", 'figures'))
os.makedirs(figures_dir, exist_ok=True)

# Saving metrics and confusion matrix
save_metrics(y_test, y_pred, y_prob[:, 1], output_dir)
save_confusion_matrix(y_test, y_pred, figures_dir)
