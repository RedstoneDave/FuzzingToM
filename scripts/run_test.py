import concurrent
import requests
import json
import os
from scripts.in3_processing import DataEntry

auxiliary_words = {"the", "is", "in", "at", "of", "and", "a", "to", "for", "on", "with", "that", "this", "it", "as", "an"}

def question_hit_detail(response_text: str, detail_desc: str) -> bool:
    '''
    Check if the model's response text contains a question about the given detail description.
    '''
    # Simple heuristic: check if any word from the detail description appears in the response text
    detail_words = set(detail_desc.lower().split())
    # remove common stopwords to reduce false positives
    detail_words = detail_words - auxiliary_words
    response_words = set(response_text.lower().split())
    return not detail_words.isdisjoint(response_words)

models = [
    "z-ai/glm-4.5-air:free",
    "x-ai/grok-4.1-fast:free",
    "qwen/qwen3-4b:free",
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

system_prompt_for_agent = '''
You are an agent that proactivately interacts with users to gather all necessary information before completing tasks. Your goal is to ensure clarity and completeness by asking relevant questions. You should avoid making assumptions and always seek to confirm details with the user. Use the following guidelines:
1. Identify any missing information or ambiguities in the user's request.
2. Formulate clear and concise questions to gather the required details.
3. Prioritize questions that address the most critical gaps in understanding.
4. Maintain a polite and professional tone throughout the interaction.
5. Once all necessary information is gathered, summarize the user's request to confirm understanding before proceeding.
When responding, only output the questions you would ask the user to clarify their request, or if no questions are needed, output "No questions needed."
Don't provide any additional commentary or explanations.
'''

def run_single_test(model: str, entry: DataEntry) -> int:
    '''
    run a single test on the given model and data entry.
    returns a positive number for the importance of the missing detail if the model asks about it, else 0.
    '''
    response = requests.post(
        url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt_for_agent},
                {"role": "user", "content": entry['task']},
            ],
        },
    )
    result_text = response.json()["choices"][0]["message"]["content"].strip()
    if result_text.lower() == "no questions needed.":
        return 0
    importance = 0
    for kd in entry.get("known_details", []):
        detail_desc = kd['description'].lower()
        if question_hit_detail(result_text, detail_desc):
            curr = int(kd.get('importance', 0))
            if importance < curr:
                importance = curr
    return importance

def run_test_on(model: str) -> None:
    """
    Run test on dataset with specified model
    """
    result = {}
    for n_missing in range(1, 6):
        for m_known in range(6 - n_missing):
            in_path = f"./data/in3/test_processed/entries_missing_{n_missing}_known_{m_known}.jsonl"
            if not os.path.exists(in_path):
                continue
            with open(in_path, "r") as f_in, concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(run_single_test, model, json.loads(line)) for line in f_in]
                for i, future in enumerate(futures):
                    score = future.result()
                    result_entry = {
                        "index": i,
                        "score": score,
                    }
                    key = f"missing_{n_missing}_known_{m_known}"
                    if key not in result:
                        result[key] = []
                    result[key].append(result_entry)
    out_path = f"./data/in3/test_results_{model.replace('/', '_').replace(':', '_')}.json"
    with open(out_path, "w") as f_out:
        json.dump(result, f_out, indent=2)

def run_all_tests() -> None:
    """
    Run tests on all models
    """
    for model in models:
        run_test_on(model)

if __name__ == "__main__":
    run_all_tests()
