import google.generativeai as genai
import os
import time
import random
from dotenv import load_dotenv
import pandas as pd
import seaborn as sns

load_dotenv(dotenv_path="../.env")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

base_model = "models/gemini-1.5-flash-001-tuning"

name = f"generate-num-{random.randint(0,10000)}"
operation = genai.create_tuned_model(
    # You can use a tuned model here too. Set `source_model="tunedModels/..."`
    source_model=base_model,
    training_data=[
        {
            "text_input": "1",
            "output": "2",
        },
        {
            "text_input": "3",
            "output": "4",
        },
        {
            "text_input": "-3",
            "output": "-2",
        },
        {
            "text_input": "twenty two",
            "output": "twenty three",
        },
        {
            "text_input": "two hundred",
            "output": "two hundred one",
        },
        {
            "text_input": "ninety nine",
            "output": "one hundred",
        },
        {
            "text_input": "8",
            "output": "9",
        },
        {
            "text_input": "-98",
            "output": "-97",
        },
        {
            "text_input": "1,000",
            "output": "1,001",
        },
        {
            "text_input": "10,100,000",
            "output": "10,100,001",
        },
        {
            "text_input": "thirteen",
            "output": "fourteen",
        },
        {
            "text_input": "eighty",
            "output": "eighty one",
        },
        {
            "text_input": "one",
            "output": "two",
        },
        {
            "text_input": "three",
            "output": "four",
        },
        {
            "text_input": "seven",
            "output": "eight",
        },
    ],
    id=name,
    epoch_count=100,
    batch_size=4,
    learning_rate=0.001,
)

for status in operation.wait_bar():
    time.sleep(10)

model = operation.result()

snapshots = pd.DataFrame(model.tuning_task.snapshots)

sns.lineplot(data=snapshots, x="epoch", y="mean_loss")

model = genai.GenerativeModel(model_name=f"tunedModels/{name}")
result = model.generate_content("four")
print(result.text)
