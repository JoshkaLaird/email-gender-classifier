import torch
import torch.nn as nn

from typing import Dict, Tuple

from core.dataset import encode_email
from core.model import EmailCharCNN


def predict_email(
    str_email: str,
    obj_model: nn.Module,
    dict_char_to_id: Dict[str, int],
    dict_label_to_id: Dict[str, int],
    int_max_length: int,
    str_device: str
) -> Tuple[str, float]:
    dict_id_to_label = {int_id: str_label for str_label, int_id in dict_label_to_id.items()}

    list_token_ids = encode_email(
        str_email=str_email,
        dict_char_to_id=dict_char_to_id,
        int_max_length=int_max_length
    )

    ts_input_ids = torch.tensor([list_token_ids], dtype=torch.long).to(str_device)

    obj_model.eval()
    with torch.no_grad():
        ts_logits = obj_model(ts_input_ids)
        ts_probs = torch.softmax(ts_logits, dim=1)
        ts_pred_id = torch.argmax(ts_probs, dim=1)

    int_pred_id = int(ts_pred_id.item())
    str_pred_label = dict_id_to_label[int_pred_id]
    float_confidence = float(ts_probs[0, int_pred_id].item())

    return str_pred_label, float_confidence


def main() -> None:
    str_model_path = r"output\training\cnn.pt"
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

    list_test_emails = [
        "max.mustermann@gmail.com",
        "anna89@web.de",
        "vogel66@gmx.de",
        "dragonkiller42@outlook.de",
        "X1992@web.de"
    ]

    for str_email in list_test_emails:
        str_label, float_confidence = predict_email(
            str_email=str_email,
            obj_model=obj_model,
            dict_char_to_id=dict_char_to_id,
            dict_label_to_id=dict_label_to_id,
            int_max_length=int_max_length,
            str_device=str_device
        )
        print(f"email: {str_email}")
        print(f"prediction: {str_label}")
        print(f"confidence: {float_confidence:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
