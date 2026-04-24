import os
import time
import asyncio
from typing import Literal, Optional

import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


os.environ["OLLAMA_NO_GPU"] = "1"


class BatchAnalysis(BaseModel):
    genders: list[Literal["male", "female", "neutral"]]


obj_provider = OpenAIProvider(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

obj_model = OpenAIChatModel(
    "qwen3.5:4b",
    provider=obj_provider,
)

obj_agent = Agent(
    model=obj_model,
    output_type=BatchAnalysis,
    retries=3,
    system_prompt="""
You are a gender classification system.

Rules:
- You receive a numbered list of entries, each with an email and firstname.
- Assume German context.
- For each entry, infer the most likely gender.
- You must return exactly one of: male, female, neutral per entry.
- If the first name is ambiguous, missing, unclear, synthetic, or uncertain, return neutral.
- Return results in the same order as the input.
- Return only the structured result.
""",
)


async def classify_batch(list_rows: list[dict]) -> list[dict]:
    float_start = time.time()

    str_prompt = "\n".join(
        f"{i + 1}. email: {row['email']}, firstname: {row['firstname']}"
        for i, row in enumerate(list_rows)
    )

    try:
        obj_result = await obj_agent.run(str_prompt)
        list_genders = obj_result.output.genders

        return [
            {
                "email": row["email"],
                "vorname": row["firstname"],
                "gender": list_genders[i] if i < len(list_genders) else "neutral",
                "success": i < len(list_genders),
                "duration": round(time.time() - float_start, 2),
            }
            for i, row in enumerate(list_rows)
        ]

    except Exception:
        return [
            {
                "email": row["email"],
                "vorname": row["firstname"],
                "gender": "neutral",
                "success": False,
                "duration": round(time.time() - float_start, 2),
            }
            for row in list_rows
        ]


async def process_csv(
    str_input_csv_path: str,
    str_output_csv_path: str,
    int_batch_size: int = 10,
    int_limit: Optional[int] = None,  # number of batches, None = all
) -> pd.DataFrame:
    df_input = pd.read_csv(str_input_csv_path)
    df_input.columns = [str_col.strip() for str_col in df_input.columns]

    if "EMAIL" not in df_input.columns or "VORNAME" not in df_input.columns:
        raise ValueError("CSV must contain the columns 'EMAIL' and 'VORNAME'.")

    df_work = df_input[["EMAIL", "VORNAME"]].copy()
    if int_limit is not None:
        df_work = df_work.head(int_limit * int_batch_size)

    df_work["EMAIL"] = df_work["EMAIL"].fillna("").astype(str).str.strip()
    df_work["VORNAME"] = df_work["VORNAME"].fillna("").astype(str).str.strip()

    list_rows = [
        {"email": row["EMAIL"], "firstname": row["VORNAME"]}
        for _, row in df_work.iterrows()
    ]

    list_batches = [
        list_rows[i:i + int_batch_size]
        for i in range(0, len(list_rows), int_batch_size)
    ]

    list_results = []
    for batch in tqdm(list_batches, desc="Klassifiziere"):
        list_results.extend(await classify_batch(batch))

    df_result = pd.DataFrame(list_results)
    df_result.to_csv(str_output_csv_path, index=False)

    return df_result


if __name__ == "__main__":

    df_result = asyncio.run(
        process_csv(
            str_input_csv_path=r"C:\Users\Hyperhaven\Downloads\random_email_dataset.csv",
            str_output_csv_path=r"C:\Users\Hyperhaven\Downloads\emails_results.csv",
            int_batch_size=100,
            int_limit=1,  # number of batches, None = all
        )
    )

    print(df_result.head())
