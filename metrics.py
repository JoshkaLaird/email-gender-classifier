import argparse
import json
import os
import pandas as pd
import torch

from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from core.dataset import encode_email
from core.model import EmailCharCNN


def predict_batch(
    list_emails: List[str],
    obj_model: EmailCharCNN,
    dict_char_to_id: Dict[str, int],
    dict_label_to_id: Dict[str, int],
    int_max_length: int,
    str_device: str,
    int_batch_size: int = 256
) -> Tuple[List[str], List[float]]:
    dict_id_to_label = {
        int_id: str_label for str_label, int_id in dict_label_to_id.items()
    }

    list_all_token_ids = [
        encode_email(
            str_email=str_email,
            dict_char_to_id=dict_char_to_id,
            int_max_length=int_max_length
        )
        for str_email in list_emails
    ]

    ts_all_input_ids = torch.tensor(list_all_token_ids, dtype=torch.long)

    list_predictions = []
    list_confidences = []

    obj_model.eval()

    with torch.no_grad():
        for int_start in range(0, len(ts_all_input_ids), int_batch_size):
            int_end = int_start + int_batch_size
            ts_batch_input_ids = ts_all_input_ids[int_start:int_end].to(str_device)

            ts_logits = obj_model(ts_batch_input_ids)
            ts_probs = torch.softmax(ts_logits, dim=1)

            ts_pred_ids = torch.argmax(ts_probs, dim=1)
            ts_pred_conf = torch.gather(
                ts_probs,
                dim=1,
                index=ts_pred_ids.unsqueeze(1)
            ).squeeze(1)

            list_predictions.extend(
                [dict_id_to_label[int(int_pred_id)] for int_pred_id in ts_pred_ids.cpu().tolist()]
            )
            list_confidences.extend(
                [float(float_conf) for float_conf in ts_pred_conf.cpu().tolist()]
            )

    return list_predictions, list_confidences


def build_error_analysis(df_eval: pd.DataFrame) -> Dict[str, object]:
    dict_output = {}

    df_errors = df_eval[df_eval["is_correct"] == False].copy()
    dict_output["int_total_rows"] = int(len(df_eval))
    dict_output["int_total_errors"] = int(len(df_errors))
    dict_output["float_error_rate"] = float((~df_eval["is_correct"]).mean())

    if "email_type" in df_eval.columns:
        df_by_type = (
            df_eval
            .groupby("email_type", dropna=False)
            .agg(
                int_count=("email", "count"),
                float_accuracy=("is_correct", "mean")
            )
            .reset_index()
            .sort_values(["float_accuracy", "int_count"], ascending=[True, False])
        )
        dict_output["list_accuracy_by_email_type"] = df_by_type.to_dict(orient="records")

    df_by_true_label = (
        df_eval
        .groupby("gender", dropna=False)
        .agg(
            int_count=("email", "count"),
            float_accuracy=("is_correct", "mean"),
            float_avg_confidence=("confidence", "mean")
        )
        .reset_index()
        .sort_values("float_accuracy", ascending=True)
    )
    dict_output["list_accuracy_by_true_label"] = df_by_true_label.to_dict(orient="records")

    list_error_cols = ["email", "gender", "prediction", "confidence"]
    if "email_type" in df_errors.columns:
        list_error_cols.append("email_type")

    df_top_errors = (
        df_errors[list_error_cols]
        .sort_values("confidence", ascending=False)
        .head(50)
    )
    dict_output["list_top_confident_errors"] = df_top_errors.to_dict(orient="records")

    return dict_output


def main() -> None:
    obj_parser = argparse.ArgumentParser()
    obj_parser.add_argument("--model", default=r"output\training\cnn.pt")
    obj_parser.add_argument("--csv", default="data/test1_synthetic.csv")
    obj_args = obj_parser.parse_args()

    str_model_path = obj_args.model
    str_input_csv = obj_args.csv

    os.makedirs(r"output\metrics", exist_ok=True)
    str_output_predictions_csv = r"output\metrics\evaluation_predictions.csv"
    str_output_confusion_csv = r"output\metrics\evaluation_confusion_matrix.csv"
    str_output_metrics_json = r"output\metrics\evaluation_metrics.json"

    str_device = "cuda" if torch.cuda.is_available() else "cpu"

    dict_checkpoint = torch.load(str_model_path, map_location=str_device)

    dict_char_to_id = dict_checkpoint["char_to_id"]
    dict_label_to_id = dict_checkpoint["label_to_id"]
    int_max_length = dict_checkpoint["max_length"]

    obj_model = EmailCharCNN(
        int_vocab_size=len(dict_char_to_id),
        int_embedding_dim=64,
        int_num_classes=len(dict_label_to_id)
    ).to(str_device)

    obj_model.load_state_dict(dict_checkpoint["model_state_dict"])

    df_input = pd.read_csv(str_input_csv)

    if "email" not in df_input.columns:
        raise ValueError("Input CSV must contain column 'email'.")

    if "gender" not in df_input.columns:
        raise ValueError("Input CSV must contain column 'gender' for evaluation.")

    list_predictions, list_confidences = predict_batch(
        list_emails=df_input["email"].astype(str).tolist(),
        obj_model=obj_model,
        dict_char_to_id=dict_char_to_id,
        dict_label_to_id=dict_label_to_id,
        int_max_length=int_max_length,
        str_device=str_device,
        int_batch_size=256
    )

    df_eval = df_input.copy()
    df_eval["prediction"] = list_predictions
    df_eval["confidence"] = list_confidences
    df_eval["is_correct"] = df_eval["gender"] == df_eval["prediction"]

    list_labels = ["man", "woman", "unknown"]

    float_accuracy = accuracy_score(
        df_eval["gender"],
        df_eval["prediction"]
    )

    dict_report = classification_report(
        df_eval["gender"],
        df_eval["prediction"],
        labels=list_labels,
        output_dict=True,
        digits=4,
        zero_division=0
    )

    arr_confusion = confusion_matrix(
        df_eval["gender"],
        df_eval["prediction"],
        labels=list_labels
    )

    df_confusion = pd.DataFrame(
        arr_confusion,
        index=[f"true_{str_label}" for str_label in list_labels],
        columns=[f"pred_{str_label}" for str_label in list_labels]
    )

    arr_precision, arr_recall, arr_f1, arr_support = precision_recall_fscore_support(
        df_eval["gender"],
        df_eval["prediction"],
        labels=list_labels,
        zero_division=0
    )

    list_per_class_metrics = []
    for int_index, str_label in enumerate(list_labels):
        dict_class_metrics = {
            "str_label": str_label,
            "float_precision": float(arr_precision[int_index]),
            "float_recall": float(arr_recall[int_index]),
            "float_f1": float(arr_f1[int_index]),
            "int_support": int(arr_support[int_index])
        }
        list_per_class_metrics.append(dict_class_metrics)

    dict_error_analysis = build_error_analysis(df_eval=df_eval)

    dict_metrics = {
        "float_accuracy": float(float_accuracy),
        "dict_classification_report": dict_report,
        "list_per_class_metrics": list_per_class_metrics,
        "dict_error_analysis": dict_error_analysis
    }

    df_eval.to_csv(str_output_predictions_csv, index=False, encoding="utf-8")
    df_confusion.to_csv(str_output_confusion_csv, encoding="utf-8")

    with open(str_output_metrics_json, "w", encoding="utf-8") as obj_file:
        json.dump(dict_metrics, obj_file, ensure_ascii=False, indent=2)

    print("\n=== OVERALL METRICS ===")
    print(f"accuracy: {float_accuracy:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            df_eval["gender"],
            df_eval["prediction"],
            labels=list_labels,
            digits=4,
            zero_division=0
        )
    )

    print("\n=== CONFUSION MATRIX ===")
    print(df_confusion)

    if "email_type" in df_eval.columns:
        print("\n=== ACCURACY BY EMAIL TYPE ===")
        df_accuracy_by_type = (
            df_eval
            .groupby("email_type", dropna=False)
            .agg(
                int_count=("email", "count"),
                float_accuracy=("is_correct", "mean")
            )
            .reset_index()
            .sort_values(["float_accuracy", "int_count"], ascending=[True, False])
        )
        print(df_accuracy_by_type.to_string(index=False))

    print("\n=== WORST 20 ERRORS BY CONFIDENCE ===")
    list_error_cols = ["email", "gender", "prediction", "confidence"]
    if "email_type" in df_eval.columns:
        list_error_cols.append("email_type")

    df_worst_errors = (
        df_eval[df_eval["is_correct"] == False]
        .sort_values("confidence", ascending=False)
        .head(20)[list_error_cols]
    )
    print(df_worst_errors.to_string(index=False))

    print(f"\nSaved: {str_output_predictions_csv}")
    print(f"Saved: {str_output_confusion_csv}")
    print(f"Saved: {str_output_metrics_json}")


if __name__ == "__main__":
    main()
