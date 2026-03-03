import numpy as np
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support

smoothing_function = SmoothingFunction().method1


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy

    Args:
    y_true: List or array of true labels
    y_pred: List or array of predicted labels

    Returns:
    float: Accuracy (proportion of correctly predicted samples)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average="macro"):
    """
    Calculate precision using sklearn's precision_recall_fscore_support

    Args:
    y_true: List or array of true labels
    y_pred: List or array of predicted labels
    average: 'macro' for macro-average precision

    Returns:
    float: Precision
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    precision, _, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=average, zero_division=0
    )

    return precision


def recall_score(y_true, y_pred, average="macro"):
    """
    Calculate recall using sklearn's precision_recall_fscore_support

    Args:
    y_true: List or array of true labels
    y_pred: List or array of predicted labels
    average: 'macro' for macro-average recall

    Returns:
    float: Recall
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    _, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=average, zero_division=0
    )

    return recall


def f1_score(y_true, y_pred, average="macro"):
    """
    Calculate F1 score using sklearn's precision_recall_fscore_support

    Args:
    y_true: List or array of true labels
    y_pred: List or array of predicted labels
    average: 'macro' for macro-average F1 score

    Returns:
    float: F1 score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=average, zero_division=0
    )

    return f1


def is_best_model(current_results, best_results, stage):
    """
    Determine whether the current model is the best model

    Args:
        current_metrics: Dictionary of current evaluation metrics
        best_metric: Historical best metric value
        stage: Training stage
    """
    if stage == "classification":
        current = current_results["classification_metrics"]["f1_macro_cli"]
    elif stage == "generation":
        current = current_results["generation_metrics"]["f1_macro_gen"]
    else:  # joint
        # For joint training, weighted combination can be used
        current = (
            current_results["classification_metrics"]["f1_macro_cli"] * 0.8
            + current_results["generation_metrics"]["f1_macro_gen"] * 0.2
        )

    return current > best_results
