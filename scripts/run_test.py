import concurrent
from concurrent.futures import as_completed, ThreadPoolExecutor, Future
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
5. If all necessary information is gathered, you should output "No questions needed".
When responding, only output the questions you would ask the user to clarify their request, with each question a line. Or, if no questions are needed, output "No questions needed."
Don't provide any additional commentary or explanations other than your clarifying questions.
Don't answer the questions or complete the task yourself.
'''

def run_single_test(model: str, entry: DataEntry, index: int) -> tuple[str, int]:
    '''
    run a single test and return the result
    '''
    try:
        result_text = _get_response(model, entry)
        print('model response:', result_text)
        return result_text, index
    except Exception as e:
        print(f"Error during model response: {e}")
        return "Error", index

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
    
    with out_path.open("r") as original_file:
        try:
            result = json.load(original_file)
        except json.JSONDecodeError:
            result = {}
    
    for n_missing in range(6):
        for m_known in range(6 - n_missing):
            in_path = data_dir / f"test_missing_{n_missing}_known_{m_known}.jsonl"
            if not in_path.exists():
                continue
            key = f"missing_{n_missing}_known_{m_known}"
            if key not in result:
                result[key] = []
            with in_path.open("r") as f_in, ThreadPoolExecutor() as executor:
                futures : list[Future] = []
                for i, line in enumerate(f_in):
                    if any(entry["index"] == i for entry in result[key]):
                        continue
                    futures.append(executor.submit(run_single_test, model, json.loads(line), i))
                for future in as_completed(futures):
                    answer, i = future.result()
                    result_entry = {
                        "index": i,
                        "answer": answer,
                    }
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
