import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel(model_name="tunedModels/generate-num-2948")
result = model.generate_content("four")
print(result.text)

print(model)
