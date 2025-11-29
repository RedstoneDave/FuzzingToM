import concurrent
from concurrent.futures import as_completed
from pathlib import Path
import dotenv
import json
import os
from scripts.in3_processing import DataEntry
import openai

models = [
    "meta-llama/llama-3.2-3b-instruct",
    "google/gemma-3-4b-it",
    # "mistralai/mistral-7b-instruct", # Wierd Outputs
    "qwen/qwen3-8b",
    # "thudm/glm-4-9b", # No Endpoint
]

system_prompt_for_agent = '''
You are an agent that proactivately interacts with users to gather all necessary information before completing tasks. Your goal is to ensure clarity and completeness by asking relevant questions. You should avoid making assumptions and always seek to confirm details with the user. Use the following guidelines:
1. Identify any missing information or ambiguities in the user's request.
2. Formulate clear and concise questions to gather the required details.
3. Prioritize questions that address the most critical gaps in understanding.
4. Maintain a polite and professional tone throughout the interaction.
5. If all necessary information is gathered, summarize the user's request to confirm understanding before proceeding.
When responding, only output one question you would ask the user to clarify their request, or if no questions are needed, output "No questions needed."
Don't provide any additional commentary or explanations. Don't answer the questions yourself.
'''

def run_single_test(model: str, entry: DataEntry) -> int:
    '''
    run a single test and return the result
    '''
    try:
        result_text = _get_response(model, entry)
        print('model response:', result_text)
        return result_text
    except Exception as e:
        print(f"Error during model response: {e}")
        return "Error"

dotenv.load_dotenv()
print(os.getenv("OPENAI_API_KEY", ""))
print(os.getenv("OPENAI_API_BASE", ""))

client = openai.OpenAI(
    base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    timeout=60,
)

def _get_response(model, entry: DataEntry):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt_for_agent},
            {"role": "user", "content": entry["task"]},
        ],
    ).choices[0].message.content

def run_test_on(model: str, data_dir: str | Path, output_dir: str | Path) -> None:
    """
    Run test on dataset with specified model
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    out_path = output_dir / f"test_results_{model.replace('/', '_').replace(':', '_')}.json"
        
    for n_missing in range(6):
        for m_known in range(6 - n_missing):
            in_path = data_dir / f"test_missing_{n_missing}_known_{m_known}.jsonl"
            if not in_path.exists():
                continue
            with in_path.open("r") as f_in, concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(run_single_test, model, json.loads(line)) for line in f_in]
                for i, future in enumerate(as_completed(futures)):
                    answer = future.result()
                    result_entry = {
                        "index": i,
                        "answer": answer,
                    }
                    key = f"missing_{n_missing}_known_{m_known}"
                    if key not in result:
                        result[key] = []
                    result[key].append(result_entry)
                    with out_path.open("w") as f_out:
                        json.dump(result, f_out, indent=2)

def run_all_tests() -> None:
    """
    Run tests on all models
    """
    for model in models:
        run_test_on(model, "./data/in3/test_processed/group_1", "./data/in3/test_results/group_1")
        run_test_on(model, "./data/in3/test_processed/group_2", "./data/in3/test_results/group_2")

if __name__ == "__main__":
    run_all_tests()
