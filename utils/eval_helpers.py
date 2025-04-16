from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils.threshold import find_threshold_binary_search

def evaluate_model_for_recall(target_class, desired_recall, y_true, y_pred_proba):
    """
    Evaluate the model by finding the threshold for a specified recall of a target class,
    and then print the classification report and confusion matrix.
    Args:
        target_class (int): The class for which the recall is specified (0 or 1).
        desired_recall (float): The desired recall value (between 0 and 1).
        y_true (list or np.array): Ground truth labels.
        y_pred_proba (list or np.array): Predicted probabilities for the positive class.

    Returns:
        float: The threshold value that achieves the desired recall.
    """
    # Sort predictions and true labels by predicted probabilities
    threshold = find_threshold_binary_search(y_true=y_true, y_pred_proba=y_pred_proba, target_class=target_class, target_recall=desired_recall)[0]

    # Convert probabilities to binary predictions using the threshold
    binary_preds = [1 if p >= threshold else 0 for p in y_pred_proba]

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, binary_preds, target_names=['Class 0', 'Class 1'], digits=4))

    # Compute and display confusion matrix
    cm = confusion_matrix(y_true, binary_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')  # Use 'd' to display integers
    plt.title("Confusion Matrix")
    plt.show()

    return threshold