import pandas as pd
import torch

from typing import Dict, List, Tuple
from torch.utils.data import Dataset


def extract_local_part(str_email: str) -> str:
    if not isinstance(str_email, str):
        return ""
    if "@" not in str_email:
        return str_email.strip().lower()
    return str_email.split("@")[0].strip().lower()


def encode_email(
    str_email: str,
    dict_char_to_id: Dict[str, int],
    int_max_length: int
) -> List[int]:
    str_local = extract_local_part(str_email)
    list_token_ids = []

    for str_char in str_local[:int_max_length]:
        int_token_id = dict_char_to_id.get(str_char, dict_char_to_id["<unk>"])
        list_token_ids.append(int_token_id)

    while len(list_token_ids) < int_max_length:
        list_token_ids.append(dict_char_to_id["<pad>"])

    return list_token_ids


def build_vocab(sr_emails: pd.Series) -> Dict[str, int]:
    set_chars = set()

    for str_email in sr_emails.astype(str):
        str_local = extract_local_part(str_email)
        for str_char in str_local:
            set_chars.add(str_char)

    list_chars = sorted(list(set_chars))
    dict_char_to_id = {"<pad>": 0, "<unk>": 1}

    for int_index, str_char in enumerate(list_chars, start=2):
        dict_char_to_id[str_char] = int_index

    return dict_char_to_id


class EmailCharDataset(Dataset):
    def __init__(
        self,
        df_data: pd.DataFrame,
        dict_char_to_id: Dict[str, int],
        dict_label_to_id: Dict[str, int],
        int_max_length: int
    ) -> None:
        self.df_data = df_data.reset_index(drop=True)
        self.dict_char_to_id = dict_char_to_id
        self.dict_label_to_id = dict_label_to_id
        self.int_max_length = int_max_length

    def __len__(self) -> int:
        return len(self.df_data)

    def _encode_text(self, str_text: str) -> List[int]:
        list_token_ids = []

        for str_char in str_text[:self.int_max_length]:
            int_token_id = self.dict_char_to_id.get(str_char, self.dict_char_to_id["<unk>"])
            list_token_ids.append(int_token_id)

        while len(list_token_ids) < self.int_max_length:
            list_token_ids.append(self.dict_char_to_id["<pad>"])

        return list_token_ids

    def __getitem__(self, int_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        str_email = self.df_data.loc[int_index, "email"]
        str_label = self.df_data.loc[int_index, "gender"]

        str_local = extract_local_part(str_email)
        list_token_ids = self._encode_text(str_local)

        ts_input_ids = torch.tensor(list_token_ids, dtype=torch.long)
        ts_label = torch.tensor(self.dict_label_to_id[str_label], dtype=torch.long)

        return ts_input_ids, ts_label
