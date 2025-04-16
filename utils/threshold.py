import numpy as np
import warnings
from sklearn.metrics import recall_score

def find_threshold_binary_search(y_true, y_pred_proba, target_recall=0.98, target_class=0, n_iterations=1000, tolerance=1e-7):
    """
    Uses binary search to find a classification threshold that aims to meet or
    exceed a target recall for a specific class.

    Finds:
     - For target_class=1: The highest threshold 't' such that recall(t) >= target_recall.
     - For target_class=0: The lowest threshold 't' such that recall(t) >= target_recall.

    Note: Due to the discrete nature of recall, the exact target value may not be
    achievable. This function finds a boundary threshold satisfying the condition.

    Args:
        y_true (np.array): Array of true binary labels (e.g., 0 or 1).
        y_pred_proba (np.array): Array of predicted probabilities for the *positive*
                                  class (class 1). Shape (n_samples,).
        target_recall (float): The desired minimum recall level (between 0.0 and 1.0).
        target_class (int): The class (0 or 1) for which the recall target
                            should be met.
        n_iterations (int): Number of iterations for binary search (controls precision).
                            Defaults to 100.
        tolerance (float): Alternative stopping criterion: stop if high - low < tolerance.
                           Defaults to 1e-7. (The loop also stops after n_iterations).

    Returns:
        tuple: A tuple containing:
            - best_threshold (float or None): The boundary threshold found by binary search.
              Returns None if the target recall is never met.
            - achieved_recall (float or None): The actual recall achieved for the
              target_class at the best_threshold.

    Raises:
        ValueError: If target_recall is not between 0 and 1, or if target_class
                    is not 0 or 1, or if input arrays are incompatible.
    """
    if not 0.0 <= target_recall <= 1.0:
        raise ValueError("target_recall must be between 0.0 and 1.0")
    if target_class not in [0, 1]:
        raise ValueError("target_class must be 0 or 1")
    if len(y_true) != len(y_pred_proba):
        raise ValueError("y_true and y_pred_proba must have the same length")
    if len(np.unique(y_true)) < 2 and len(y_true) > 0 :
         warnings.warn("Warning: y_true contains only one class. Recall might be 0 or 1 trivially.")
    elif len(np.unique(y_true)) > 2:
         warnings.warn("Warning: This function is designed for binary classification, "
                       "but y_true contains more than two unique values.")


    low = 0.0
    high = 1.0
    best_threshold = None # Use None to indicate target never met

    for _ in range(n_iterations):
        if high - low < tolerance:
            break

        mid = low + (high - low) / 2 # More stable than (low+high)/2 for large numbers

        # Predict using the midpoint threshold
        y_pred = (y_pred_proba >= mid).astype(int)
        current_recall = recall_score(y_true, y_pred, pos_label=target_class, zero_division=0)

        # --- Binary Search Logic ---
        if target_class == 1:
            # Recall_1 decreases as threshold increases. We want highest threshold >= target.
            if current_recall >= target_recall:
                # This threshold works, maybe a higher one also works?
                best_threshold = mid # Store this as a potential best
                low = mid # Search in the higher half (thresholds >= mid)
            else:
                # Recall is too low, threshold is too high. Need to lower threshold.
                high = mid # Search in the lower half (thresholds < mid)

        else: # target_class == 0
            # Recall_0 increases as threshold increases. We want lowest threshold >= target.
             if current_recall >= target_recall:
                # This threshold works, maybe a lower one also works?
                best_threshold = mid # Store this as a potential best
                high = mid # Search in the lower half (thresholds <= mid)
             else:
                # Recall is too low, threshold is too low. Need to increase threshold.
                low = mid # Search in the higher half (thresholds > mid)


    # --- Final Result ---
    final_recall = None
    if best_threshold is not None:
        # Recalculate recall at the final best_threshold for confirmation
        y_pred_final = (y_pred_proba >= best_threshold).astype(int)
        final_recall = recall_score(y_true, y_pred_final, pos_label=target_class, zero_division=0)

        # It's possible the final recall slightly dropped below target due to float precision
        # or boundary effects in the last step. Usually minor.
        print(f"Target Recall: >= {target_recall:.4f} for Class {target_class}")
        print(f"Threshold found by Binary Search: {best_threshold:.7f}") # Higher precision
        print(f"Achieved Recall at Threshold: {final_recall:.4f}")
        if final_recall < target_recall:
             warnings.warn(f"Note: Final recall {final_recall:.4f} is slightly below target {target_recall:.4f}."
                           " This can happen due to float precision near the decision boundary.")

    else:
        warnings.warn(f"Target recall {target_recall} for class {target_class} was not met at any threshold.")
        # Optional: Could do a separate search for the absolute closest recall here if needed.
        # For now, just return None as per the function's goal.
        print(f"Target Recall: >= {target_recall:.4f} for Class {target_class}")
        print(f"Result: Target recall was not achievable.")


    return best_threshold, final_recall