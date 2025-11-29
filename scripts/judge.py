from ast import Tuple
from enum import StrEnum
import os
from pathlib import Path
import random
import re
from time import sleep
from typing import TypedDict
import dotenv
import dashscope
import json

from scripts.in3_processing import DataEntry

dotenv.load_dotenv()

judge_prompt = """
You are an automated judge. Given a single description and one or more questions, decide whether any question asks about that description.

Rules:
- If at least one question is relevant to the description, output exactly: yes
- If none are relevant, output exactly: no
- If the input contains no question (e.g., plain text, logs, code, or truncated content with no question), output exactly: error

Relevance guidance:
- A question is relevant if it directly asks for information about the description or asks for details that correspond to it. Paraphrases and synonyms count.
- Implied or follow-up questions that clearly target the described attribute also count as relevant.
- Questions about unrelated topics do not count.

Do not output anything else—no punctuation, explanation, or extra text. Examples:
Description: "Cuisine preference"
Question: "What type of food do you like?"
-> yes

Description: "Preferred travel destination"
Question: "What is your favorite color?"
-> no

Description: "Programming Language"
Question: "Here are some unit tests... (truncated)"
-> error

Here is the description and the question:
"""

def judge_question_single(description: str, question: str) -> bool | None:
    '''
    use dashscope to judge whether the question hit the missing detail
    '''
    while True:
        response = dashscope.Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            model="qwen-max-latest",
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": f"Description: {description}\nQuestions:\n{question}"}
            ],
        )
        if response.status_code == 429:
            print('rate limited, sleeping...')
            sleep(random.uniform(30, 60))
            continue
        elif response.status_code != 200:
            print('Error')
            return None
        text = response.output.text.lower().strip()
        error = "error" in text
        if error:
            print('non-question detected')
            print(description, question, text, sep='\n-----\n')
            return None
        yes = "yes" in text
        no = "no" in text
        if yes and no:
            print('What? Why yes and no?')
            print(description, question, text, sep='\n-----\n')
            return False
        if not (yes or no):
            print('What? Why neither yes nor no?')
            print(description, question, text, sep='\n-----\n')
            return False
        return yes

path_data = Path("data/in3/test_processed/")
path_results = Path("data/in3/test_results/")
path_output = Path("data/in3/test_judged/")


def judge_question(question: str, entry: DataEntry) -> tuple[int, bool]:
    '''
    judge the question against the data entry
    returns a tuple of (number of hits, questioned or not)
    '''
    if question.strip().lower() == 'no questions needed.':
        return 0, False
    if not entry['vague']:
        return 0, True
    count = 0
    for md in entry.get('missing_details', []):
        res = judge_question_single(md['description'], question)
        if res:
            count += 1
        if res is None:
            return 0, False
    return count, True

def judge(group_name: str, model_name: str) -> None:
    out_dir = path_output / group_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name_modified = model_name.replace('/', '_').replace(':', '_')
    in_path = path_results / group_name / f"test_results_{model_name_modified}.json"
    out_path = out_dir / f"judged_results_{model_name_modified}.json"
    with in_path.open("r") as f_in:
        results = json.load(f_in)
    judged_results = {}
    out_path.touch()
    with out_path.open('r') as original_file:
        try:
            judged_results = json.load(original_file)
        except json.JSONDecodeError:
            judged_results = {}
    for key, entries in results.items():
        regex = re.compile(r"missing_(\w+)_known_(\w+)")
        match = regex.match(key)
        n_missing, m_known = map(int, match.groups())
        if key not in judged_results:
            judged_results[key] = judged_entries = []
        else:
            judged_entries = judged_results[key]
        data_in_path = path_data / group_name / f"test_missing_{n_missing}_known_{m_known}.jsonl"
        with data_in_path.open("r") as f_data_in:
            data_entries = [json.loads(line) for line in f_data_in]
        for result_entry in entries:
            if any(je['index'] == result_entry['index'] for je in judged_entries):
                continue
            index = result_entry['index']
            answer = result_entry['answer']
            data_entry = data_entries[index]
            judge_result = judge_question(answer, data_entry)
            judged_entries.append({
                "index": index,
                "answer": answer,
                "hit_count": judge_result[0],
                "do_question": judge_result[1],
            })
            with out_path.open("w") as f_out:
                json.dump(judged_results, f_out, indent=2)

groups = ["group_1", "group_2"]
models = [
    "meta-llama/llama-3.2-3b-instruct",
    "google/gemma-3-4b-it",
    # "mistralai/mistral-7b-instruct", # Wierd Outputs
    "qwen/qwen3-8b",
    # "thudm/glm-4-9b", # No Endpoint
]

if __name__ == "__main__":
    for group in groups:
        for model in models:
            print(f"Judging {group} with model {model}...")
            judge(group, model)