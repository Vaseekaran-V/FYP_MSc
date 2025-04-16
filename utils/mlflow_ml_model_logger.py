from utils.threshold import find_threshold_binary_search
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import json
import pickle


def log_classification_model(model, X_train, y_train, X_val, y_val,
                             threshold_func=find_threshold_binary_search, # Use the desired function
                             target_recall=0.98,
                             target_class=0,
                             run_name="ML Model Run"):
    """
    Trains, evaluates (using potential custom threshold), logs parameters,
    metrics (to 4 decimals), confusion matrix plot (displayed and logged),
    and the model using MLflow.

    Args:
        model: Classification model instance.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        threshold_func (callable, optional): Function to find threshold.
            Defaults to find_threshold_binary_search. Set to None for 0.5 threshold.
        target_recall (float): Target recall for threshold_func.
        target_class (int): Target class for threshold_func.
        run_name (str): Name for the MLflow run.
    """
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"Starting MLflow Run: {run.info.run_name} ({run_id})")

        # Log model type and parameters (parameter formatting usually kept raw)
        mlflow.log_param("model_type", type(model).__name__)
        try:
            mlflow.log_params(model.get_params())
            print("Logged model parameters.")
        except Exception as e:
            print(f"Warning: Could not log all model parameters: {e}")
            try: # Fallback logging
                params = model.get_params()
                for key, value in params.items():
                     mlflow.log_param(key, str(value)) # Log as string if complex
                print("Logged individual model parameters (simplified).")
            except Exception as e_inner:
                 print(f"Warning: Could not log individual model parameters: {e_inner}")

        # --- Train the model ---
        print("Training the model...")
        model.fit(X_train, y_train)
        print("Model training complete.")

        # --- Evaluate on Validation Set ---
        print("Evaluating the model on the validation set...")
        try:
            y_prob_val = model.predict_proba(X_val)[:, 1]
        except Exception as e:
            print(f"Error: Could not get probabilities from model: {e}")
            mlflow.log_param("evaluation_error", "Could not get probabilities")
            return

        # --- Determine Threshold and Predictions ---
        final_threshold = 0.5
        threshold_details = "Standard 0.5 threshold"
        achieved_recall_at_threshold = None

        if threshold_func:
             # (Ensure threshold function prints are formatted to .4f inside the function itself)
             print(f"Attempting to find custom threshold using {threshold_func.__name__} "
                   f"for Recall[{target_class}] >= {target_recall:.4f}") # Use .4f
             try:
                 found_threshold, achieved_recall = threshold_func(
                     y_val, y_prob_val, target_recall=target_recall, target_class=target_class
                 )
                 if found_threshold is not None:
                     final_threshold = found_threshold
                     achieved_recall_at_threshold = achieved_recall
                     # Format the detail string here
                     recall_str = f"{achieved_recall:.4f}" if achieved_recall is not None else "N/A"
                     threshold_details = (f"Custom threshold {final_threshold:.4f} aiming for " # Use .4f
                                          f"Recall[{target_class}] >= {target_recall:.4f} " # Use .4f
                                          f"(func achieved {recall_str})")
                     mlflow.log_param("custom_threshold_function", threshold_func.__name__)
                     mlflow.log_param("custom_threshold_target_recall", target_recall)
                     mlflow.log_param("custom_threshold_target_class", target_class)
                     # Log threshold param as raw value for precision
                     mlflow.log_param("custom_threshold_value", final_threshold)
                     if achieved_recall is not None:
                         # Log metric rounded
                         mlflow.log_metric("custom_threshold_search_achieved_recall", round(achieved_recall, 4))
                 else:
                      threshold_details = (f"Custom threshold search failed (target Recall[{target_class}] >= {target_recall:.4f}), using 0.5.") # Use .4f
                      warnings.warn(threshold_details)
             except Exception as e:
                 threshold_details = f"Error during custom threshold search ({threshold_func.__name__}): {e}. Using 0.5."
                 warnings.warn(threshold_details)
                 mlflow.log_param("custom_threshold_error", str(e))
        else:
            print("No custom threshold function provided, using default 0.5.")
            mlflow.log_param("custom_threshold_value", final_threshold) # Log default

        print(f"Using threshold: {threshold_details}")
        # Log the actual threshold used for evaluation (raw value)
        mlflow.log_param("final_evaluation_threshold", final_threshold)

        y_pred_val = (y_prob_val >= final_threshold).astype(int)

        # --- Calculate Metrics (Format output to 4 decimals) ---
        print("Calculating performance metrics...")
        report = None
        try:
            accuracy = accuracy_score(y_val, y_pred_val)
            weighted_f1 = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)
            macro_f1 = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
            report = classification_report(y_val, y_pred_val, output_dict=True, zero_division=0)

            recall_class_0 = report.get('0', {}).get('recall', 0.0)
            recall_class_1 = report.get('1', {}).get('recall', 0.0)
            precision_class_0 = report.get('0', {}).get('precision', 0.0)
            precision_class_1 = report.get('1', {}).get('precision', 0.0)

            # Print metrics formatted to .4f
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(f"Validation Weighted F1-Score: {weighted_f1:.4f}")
            print(f"Validation Macro F1-Score: {macro_f1:.4f}")
            print(f"Validation Recall Class 0: {recall_class_0:.4f}")
            print(f"Validation Recall Class 1: {recall_class_1:.4f}")
            print(f"Validation Precision Class 0: {precision_class_0:.4f}")
            print(f"Validation Precision Class 1: {precision_class_1:.4f}")
            # Print classification report formatted to 4 digits
            print("\n" + classification_report(y_val, y_pred_val, digits=4, zero_division=0))

            # Log metrics rounded to 4 decimals
            mlflow.log_metric("val_accuracy", round(accuracy, 4))
            mlflow.log_metric("val_weighted_f1", round(weighted_f1, 4))
            mlflow.log_metric("val_macro_f1", round(macro_f1, 4))
            mlflow.log_metric("val_recall_class_0", round(recall_class_0, 4))
            mlflow.log_metric("val_recall_class_1", round(recall_class_1, 4))
            mlflow.log_metric("val_precision_class_0", round(precision_class_0, 4))
            mlflow.log_metric("val_precision_class_1", round(precision_class_1, 4))
            print("Logged validation metrics (rounded to 4 decimals).")

        except Exception as e:
            print(f"Error calculating or logging metrics: {e}")
            mlflow.log_param("metrics_error", str(e))

        # --- Generate, Display, Save, and Log Confusion Matrix Plot ---
        print("Generating Confusion Matrix plot...")
        fig = None
        try:
            cm = confusion_matrix(y_val, y_pred_val)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title(f'Confusion Matrix (Threshold={final_threshold:.4f})') # Format title
            labels = sorted(np.unique(y_val))
            if len(labels) >= 2 : # Basic check for labels
                 # Use string labels if needed, default assumes 0, 1, ...
                 str_labels = [str(l) for l in labels]
                 # Ensure heatmap labels match actual classes present
                 if cm.shape[0] == len(str_labels): # Check dimensions match
                      ax.xaxis.set_ticklabels(str_labels)
                      ax.yaxis.set_ticklabels(str_labels)
                 else:
                     print(f"Warning: CM shape {cm.shape} doesn't match number of labels {len(str_labels)}. Using default ticks.")

            plt.tight_layout()

            # Save
            cm_plot_path = "confusion_matrix_val.png"
            plt.savefig(cm_plot_path)
            print(f"Saved Confusion Matrix plot to: {cm_plot_path}")

            # Display *** NEW ***
            plt.show()
            print("Displayed Confusion Matrix plot.")

            # Log
            mlflow.log_artifact(cm_plot_path)
            print(f"Logged Confusion Matrix plot artifact.")

        except Exception as e:
            print(f"Warning: Could not generate, display, or log Confusion Matrix plot: {e}")
            mlflow.log_param("confusion_matrix_plot_error", str(e))
            # Fallback logging raw CM numbers
            try:
                if 'cm' in locals() and isinstance(cm, np.ndarray) and cm.size == 4:
                     tn, fp, fn, tp = cm.ravel()
                     mlflow.log_metric("cm_val_TN", int(tn)) # Log as int
                     mlflow.log_metric("cm_val_FP", int(fp))
                     mlflow.log_metric("cm_val_FN", int(fn))
                     mlflow.log_metric("cm_val_TP", int(tp))
                     print("Logged Confusion Matrix numbers as metrics (fallback).")
            except Exception as e_cm_log:
                 print(f"Warning: Could not log CM numbers as metrics: {e_cm_log}")
        finally:
             if fig:
                 plt.close(fig) # Close after showing and saving

        # --- Log Classification Report Artifact ---
        if report:
            print("Logging classification report artifact...")
            try:
                report_path = "classification_report_val.json"
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=4)
                mlflow.log_artifact(report_path)
                print(f"Logged classification report artifact: {report_path}")
            except Exception as e:
                print(f"Warning: Could not log classification report artifact: {e}")

        # --- Log the model ---
        print("Logging the model...")
        # ... (rest of model logging code remains the same) ...
        try:
             mlflow.sklearn.log_model(model, "model")
             print("Logged the trained model using mlflow.sklearn.")
        except Exception as e:
             print(f"Error logging the model with mlflow.sklearn: {e}")
             try:
                 model_path = "model.pkl"
                 with open(model_path, 'wb') as f: pickle.dump(model, f)
                 mlflow.log_artifact(model_path, artifact_path="model_pickle")
                 print("Logged the model as a pickle artifact (fallback).")
             except Exception as e_pickle:
                  print(f"Error: Could not log the model as pickle artifact: {e_pickle}")


        print(f"MLflow Run completed: {run_id}")
        print("View the run in the MLflow UI.")