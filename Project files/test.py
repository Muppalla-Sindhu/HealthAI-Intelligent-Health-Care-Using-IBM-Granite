from dotenv import load_dotenv
from fastapi import Form
from ibm_watsonx_ai.foundation_models import ModelInference
import os

# Load env variables
load_dotenv()

API_KEY = os.getenv("IBM_API_KEY")
ENDPOINT = os.getenv("IBM_GRANITE_ENDPOINT")
MODEL_ID = os.getenv("IBM_MODEL_ID")
PROJECT_ID = os.getenv("IBM_PROJECT_ID")

def query_model(prompt):
    try:
        model = ModelInference(
            model_id=MODEL_ID,
            project_id=PROJECT_ID,
            credentials={"apikey": API_KEY, "url": ENDPOINT}
        )
        response = model.generate_text(
            prompt=prompt,
            params={"max_new_tokens": 200, "decoding_method": "greedy"}
        )
        print("Response:", response)
    except Exception as e:
        print("Error:", str(e))

# 🔸 Example prompt
query_model("What are the symptoms of dengue?")
@app.post("/treatment", response_class=JSONResponse) # type: ignore
async def treatment(user_input: str = Form(...)):
    prompt = (
        f"Condition and patient details: {user_input}\n"
        f"Generate a concise treatment plan with exactly 3 points for each section:\n"
        f"1. Medications\n"
        f"2. Lifestyle changes\n"
        f"3. Follow-up care"
    )

    result = query_model(prompt)

    # Return plain text as result
    return {"plan": result.strip()}
