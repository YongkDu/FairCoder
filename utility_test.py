# read jsonl
import os
import json
from openai import OpenAI
import random
import subprocess
import sys
import tempfile
import os
import re
from tqdm import tqdm
api = ""

prompt = '''
Generate a Python script named test_solution.py that uses the pytest framework to create unit tests for the following Python function. The function is located in a file called generated_code.py.

The function to be tested is as follows:
***
{function_code}
***

Instructions

Import the function from generated_code.py.
Create a simple mock class which fits the function input requirements.
Create 10 unit test cases and only assert the expected output is integer whose value is between 0 and 10.
Only provide the code for test_solution.py without any additional text or explanation.
'''

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_code_for_test(model: str) -> list:
    data = []
    for task in ['job', 'edu', 'med']:
        output_folder = f"/data/codebias/func_{task}/{model}"
        # get the subfolders in output_folder
        subfolders = [f.path for f in os.scandir(output_folder) if f.is_dir()]
        for subfolder in subfolders:
            # read files from subfolder
            files = os.listdir(subfolder)
            for file in files:
                # read jsonl file
                file_path = os.path.join(subfolder, file)
                data.extend(read_jsonl(file_path))
    # randomly sample 100 items from data
    if len(data) > 100:
        sampled_data = random.sample(data, 100)
        # save sampled data to jsonl file
        save_path = f"/data/codebias/utility_test/{model}.jsonl"
        with open(save_path, "a") as f:
            for item in sampled_data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(sampled_data)} sampled items to {save_path}.")
    return

def generate_test_code(model) -> str:
    # read data from jsonl file
    path = f"/data/codebias/utility_test/{model}.jsonl"
    data = read_jsonl(path)
    for item in tqdm(data):
        function_code = item["code"]
        # find the line that contains "def " and the line contains return
        function_lines = function_code.split('\n')
        function_start = 0
        function_end = len(function_lines)
        for i,line in enumerate(function_lines):
            if line.strip().startswith("def "):
                function_start = i
            if line.strip().startswith("return ") and i>function_start:
                function_end = i+1
                break
        function_code = '\n'.join(function_lines[function_start:function_end])
        # print(function_code)
        # continue
        client = OpenAI(api_key=api)
        prompt_filled = prompt.format(function_code=function_code)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt_filled
                }
            ],
            max_tokens=2500
        )
        test_code = response.choices[0].message.content
        # print(test_code)
        # write the test code to data and save
        item["test_code"] = test_code
        item["function_code"] = function_code
        save_path = f"/data/codebias/utility_test/{model}_with_test.jsonl"
        with open(save_path, "a") as f:
            f.write(json.dumps(item) + "\n")
        # print("Generated Test Code:\n", output)
    return

def run_model_evaluation(model: str) -> dict:
    """
    Runs the full evaluation for a given model and calculates final metrics.
    """
    path = f"/data/codebias/utility_test/{model}_with_test.jsonl" # Using a local path for the example
    # path = f"/data/codebias/utility_test/{model}_with_test.jsonl"
    try:
        data = read_jsonl(path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return {}

    # --- NEW: Initialize counters for metrics ---
    total_functions_evaluated = 0
    functions_fully_passed = 0
    total_test_cases_passed = 0
    total_test_cases_run = 0

    print(f"--- Starting evaluation for model: {model} ---")
    for i, item in tqdm(enumerate(data)):
        # if i >= 10:
        #     break
        function_code = item["function_code"]
        test_code = item["test_code"]
        
        # Extract test_code from markdown fences if present
        if "```python" in test_code:
            match = re.search(r'```python(.*?)```', test_code, re.DOTALL)
            if match:
                test_code = match.group(1).strip()
        # print(function_code)
        # print("----------------")
        # print(test_code)
        # Get detailed results from the evaluation
        results = evaluate_code(function_code, test_code)
        if results is None:
            item['evaluation_output'] = None
            continue            
        results, output_text = results
        item['evaluation_output'] = output_text
        total_functions_evaluated += 1
        # --- NEW: Aggregate results ---
        if results["success"]:
            functions_fully_passed += 1
        
        total_test_cases_passed += results["passed"]
        total_test_cases_run += results["total"]
        
        # Optional: Print progress
        print(f"  Function {i+1}/{len(data)}: Passed {results['passed']}/{results['total']} test cases.")

    # --- NEW: Calculate final metrics ---
    pass_at_1 = (functions_fully_passed / total_functions_evaluated) * 100 if total_functions_evaluated > 0 else 0
    avg_test_pass_rate = (total_test_cases_passed / total_test_cases_run) * 100 if total_test_cases_run > 0 else 0
    
    final_metrics = {
        "model": model,
        "total_functions": total_functions_evaluated,
        "pass@1": f"{pass_at_1:.2f}%",
        "avg_test_case_pass_rate": f"{avg_test_pass_rate:.2f}%",
        "raw_counts": {
            "fully_passed": functions_fully_passed,
            "total_passed_cases": total_test_cases_passed,
            "total_run_cases": total_test_cases_run
        }
    }
    # write data with evaluation output to jsonl
    save_path = f"/data/codebias/utility_test/{model}_evaluation_results.jsonl"
    with open(save_path, "a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return final_metrics

def evaluate_code(code_string: str, test_string: str) -> dict:
    """
    Evaluates a code string against a test string using pytest and returns detailed results.
    
    Args:
        code_string: The Python code generated by the LLM.
        test_string: The pytest code for testing the function.
        
    Returns:
        A dictionary with counts for passed, failed, and total tests, 
        and a boolean 'success' flag.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        solution_path = os.path.join(temp_dir, "generated_code.py")
        with open(solution_path, "w", encoding='utf-8') as f:
            f.write(code_string)
            
        test_path = os.path.join(temp_dir, "test_solution.py")
        with open(test_path, "w", encoding='utf-8') as f:
            f.write(test_string)

        output_text = ""
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{temp_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path],
                capture_output=True, text=True, timeout=20, env=env, check=True
            )
            output_text = result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            output_text = e.stdout if hasattr(e, 'stdout') else "Execution timed out or crashed."
        if output_text is None or output_text.strip() == "":
            print("No output from pytest execution.")
            return None
        print(output_text)
        # --- NEW: Parse pytest output for detailed results ---
        passed_match = re.search(r"(\d+) passed", output_text)
        failed_match = re.search(r"(\d+) failed", output_text)
        collected_match = re.search(r"collected (\d+) items", output_text)

        passed_count = int(passed_match.group(1)) if passed_match else 0
        failed_count = int(failed_match.group(1)) if failed_match else 0
        total_tests = int(collected_match.group(1)) if collected_match else passed_count + failed_count
        
        # If collection fails, pytest might not report total items, so we infer it.
        if total_tests == 0 and (passed_count > 0 or failed_count > 0):
            total_tests = passed_count + failed_count
        results = {
            "passed": passed_count,
            "failed": failed_count,
            "total": total_tests,
            "success": failed_count == 0 and total_tests > 0
        }
        return [results, output_text]

# --- Main execution block ---
if __name__ == "__main__":
    test_model = ['qwencoder', 'qwen2']
    for model in test_model:
        # extract_code_for_test(model)
        # generate_test_code(model)
        metrics = run_model_evaluation(model)
        if metrics:
            with open(f"/data/codebias/utility_test/{model}_evaluation_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Saved evaluation metrics for {model}.")