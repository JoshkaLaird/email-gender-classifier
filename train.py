import random
import torch
import torch.nn as nn
import pandas as pd
import os

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from core.dataset import EmailCharDataset, build_vocab
from core.model import EmailCharCNN
from core.trainer import train_one_epoch, evaluate


def main() -> None:
    str_input_csv = "data/train.csv"
    int_max_length = 32
    int_batch_size = 256
    int_num_epochs = 8
    float_learning_rate = 1e-3
    int_random_state = 42

    random.seed(int_random_state)
    torch.manual_seed(int_random_state)

    str_device = "cuda" if torch.cuda.is_available() else "cpu"

    df_data = pd.read_csv(str_input_csv)
    df_train, df_test = train_test_split(
        df_data,
        test_size=0.2,
        random_state=int_random_state,
        stratify=df_data["gender"]
    )

    dict_char_to_id = build_vocab(df_train["email"])

    dict_label_to_id = {
        "man": 0,
        "woman": 1,
        "unknown": 2
    }

    dict_id_to_label = {int_id: str_label for str_label, int_id in dict_label_to_id.items()}

    obj_train_dataset = EmailCharDataset(
        df_data=df_train,
        dict_char_to_id=dict_char_to_id,
        dict_label_to_id=dict_label_to_id,
        int_max_length=int_max_length
    )

    obj_test_dataset = EmailCharDataset(
        df_data=df_test,
        dict_char_to_id=dict_char_to_id,
        dict_label_to_id=dict_label_to_id,
        int_max_length=int_max_length
    )

    obj_train_loader = DataLoader(obj_train_dataset, batch_size=int_batch_size, shuffle=True)
    obj_test_loader = DataLoader(obj_test_dataset, batch_size=int_batch_size, shuffle=False)

    obj_model = EmailCharCNN(
        int_vocab_size=len(dict_char_to_id),
        int_embedding_dim=64,
        int_num_classes=len(dict_label_to_id)
    ).to(str_device)

    obj_optimizer = torch.optim.Adam(obj_model.parameters(), lr=float_learning_rate)
    obj_loss_fn = nn.CrossEntropyLoss()

    for int_epoch in range(int_num_epochs):
        float_train_loss = train_one_epoch(
            obj_model=obj_model,
            obj_loader=obj_train_loader,
            obj_optimizer=obj_optimizer,
            obj_loss_fn=obj_loss_fn,
            str_device=str_device
        )

        list_true, list_pred = evaluate(
            obj_model=obj_model,
            obj_loader=obj_test_loader,
            str_device=str_device
        )

        float_accuracy = accuracy_score(list_true, list_pred)
        print(f"epoch={int_epoch + 1} train_loss={float_train_loss:.4f} test_accuracy={float_accuracy:.4f}")

    list_true, list_pred = evaluate(
        obj_model=obj_model,
        obj_loader=obj_test_loader,
        str_device=str_device
    )

    list_target_names = [dict_id_to_label[int_index] for int_index in range(len(dict_id_to_label))]
    str_report = classification_report(list_true, list_pred, target_names=list_target_names, digits=4)

    print("\n=== Classification Report ===")
    print(str_report)

    os.makedirs(r"output\training", exist_ok=True)
    torch.save(
        {
            "model_state_dict": obj_model.state_dict(),
            "char_to_id": dict_char_to_id,
            "label_to_id": dict_label_to_id,
            "max_length": int_max_length
        },
        r"output\training\cnn.pt"
    )

    with open(r"output\training\report.txt", "w", encoding="utf-8") as obj_file:
        obj_file.write(str_report)


if __name__ == "__main__":
    main()
