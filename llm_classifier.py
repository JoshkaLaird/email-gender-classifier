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

class EmailAnalysis(BaseModel):
    str_gender: Literal["male", "female", "neutral"]


obj_provider = OpenAIProvider(
    base_url="http://localhost:11434/v1",
    api_key= "ollama",
)

obj_model = OpenAIChatModel(
    "qwen3.5:4b",
    provider=obj_provider,
)

obj_agent = Agent(
    model=obj_model,
    output_type=EmailAnalysis,
    retries=3,
    system_prompt="""
You are a gender classification system.

Rules:
- You receive exactly two fields: email and firstname.
- Use only these two fields for your decision.
- Assume German context.
- Infer the most likely gender from the available information.
- You must return exactly one of: male, female, neutral.
- If the first name is ambiguous, missing, unclear, synthetic, or uncertain, return neutral.
- Return only the structured result.
""",
)


async def classify_row(str_email: str, str_firstname: str) -> dict:
    float_start = time.time()

    str_prompt = (
        f"email: {str_email}\n"
        f"firstname: {str_firstname}"
    )

    try:
        obj_result = await obj_agent.run(str_prompt)
        obj_output = obj_result.output

        dict_result = {
            "email": str_email,
            "vorname": str_firstname,
            "gender": obj_output.str_gender,
            "success": True,
            "duration": round(time.time() - float_start, 2),
        }

    except Exception:
        dict_result = {
            "email": str_email,
            "vorname": str_firstname,
            "gender": "neutral",
            "success": False,
            "duration": round(time.time() - float_start, 2),
        }

    return dict_result


async def process_csv(
    str_input_csv_path: str,
    str_output_csv_path: str,
    int_limit: Optional[int] = None,
) -> pd.DataFrame:
    df_input = pd.read_csv(str_input_csv_path)

    df_input.columns = [str_col.strip() for str_col in df_input.columns]

    if "EMAIL" not in df_input.columns or "VORNAME" not in df_input.columns:
        raise ValueError("CSV must contain the columns 'EMAIL' and 'VORNAME'.")

    df_work = df_input[["EMAIL", "VORNAME"]].copy()
    if int_limit is not None:
        df_work = df_work.head(int_limit)

    df_work["EMAIL"] = df_work["EMAIL"].fillna("").astype(str).str.strip()
    df_work["VORNAME"] = df_work["VORNAME"].fillna("").astype(str).str.strip()

    list_results = []
    for _, obj_row in tqdm(df_work.iterrows(), total=len(df_work), desc="Klassifiziere"):
        list_results.append(await classify_row(
            str_email=obj_row["EMAIL"],
            str_firstname=obj_row["VORNAME"],
        ))

    df_result = pd.DataFrame(list_results)
    df_result.to_csv(str_output_csv_path, index=False)

    return df_result


if __name__ == "__main__":

    df_result = asyncio.run(
        process_csv(
            str_input_csv_path= r"Examples\emails.csv",
            str_output_csv_path= r"Examples\emails_results.csv",
            int_limit=5, # Set to None to process the entire CSV!
        )
    )

    print(df_result.head())