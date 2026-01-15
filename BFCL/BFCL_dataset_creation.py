import json
import random
import os
from datasets import load_dataset

def extract_and_split_arrow(dataset_path, output_train_path, output_test_path, split_ratio=0.8):
    """
    Loads an Arrow dataset, parses JSON fields, extracts ground truth,
    and saves the split results into JSON files.
    """
    print(f"Loading Arrow dataset from folder: {dataset_path}")

    # Load the specific arrow file from the directory
    # Ensure the filename matches your actual data file
    ds = load_dataset(
        "arrow",
        data_files={"train": os.path.join(dataset_path, "data-00000-of-00001.arrow")}
    )["train"]

    print(f"Dataset loaded. Number of examples: {len(ds)}")

    new_examples = []

    for example in ds:

        # -------------------------------
        # PARSE fields serialized as JSON strings
        # -------------------------------
        question = example["question"]
        answer = example["answer"]
        prompt = example["prompt"]

        if isinstance(question, str):
            question = json.loads(question)

        if isinstance(answer, str):
            answer = json.loads(answer)

        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        # -------------------------------
        # Extract the first request and response
        # -------------------------------
        first_request = question[0][0]["content"]
        first_response = answer[0]  # Can be a string, dict, or list

        # -------------------------------
        # Transformation into ground_truth
        # -------------------------------
        ground_truth = []

        if isinstance(first_response, list):
            for item in first_response:
                if isinstance(item, str):
                    ground_truth.append(item)
                elif isinstance(item, dict):
                    name = item.get("name", "")
                    params = item.get("parameters", {})
                    param_str_list = []
                    for k, v in params.items():
                        if isinstance(v, str):
                            param_str_list.append(f"{k}='{v}'")
                        else:
                            param_str_list.append(f"{k}={v}")
                    param_str = ",".join(param_str_list)
                    ground_truth.append(f"{name}({param_str})")
                else:
                    ground_truth.append(str(item))
        elif isinstance(first_response, str):
            ground_truth.append(first_response)
        elif isinstance(first_response, dict):
            name = first_response.get("name", "")
            params = first_response.get("parameters", {})
            param_str_list = []
            for k, v in params.items():
                if isinstance(v, str):
                    param_str_list.append(f"{k}='{v}'")
                else:
                    param_str_list.append(f"{k}={v}")
            param_str = ",".join(param_str_list)
            ground_truth.append(f"{name}({param_str})")
        else:
            ground_truth.append(str(first_response))

        # -------------------------------
        # Construction of the final example
        # -------------------------------
        new_entry = {
            "query": first_request,
            "initial_config": example.get("initial_config", "{}"),
            "involved_classes": example.get("involved_classes", []),
            "ground_truth": ground_truth
        }

        new_examples.append(new_entry)

    # -------------------------------
    # Shuffle and Train/Test split
    # -------------------------------
    random.shuffle(new_examples)
    split_idx = int(len(new_examples) * split_ratio)

    train_examples = new_examples[:split_idx]
    test_examples = new_examples[split_idx:]

    # -------------------------------
    # Saving JSON files
    # -------------------------------
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)

    with open(output_train_path, "w", encoding="utf-8") as f:
        json.dump(train_examples, f, indent=4, ensure_ascii=False)

    with open(output_test_path, "w", encoding="utf-8") as f:
        json.dump(test_examples, f, indent=4, ensure_ascii=False)

    print(f"Train dataset saved: {output_train_path} ({len(train_examples)} examples)")
    print(f"Test dataset saved: {output_test_path} ({len(test_examples)} examples)")


if __name__ == "__main__":

    # Replace these with your actual local paths
    input_folder = "./path/to/BFCL_dataset"
    output_train_file = "./dataset/train.json"
    output_test_file = "./dataset/test.json"

    extract_and_split_arrow(input_folder, output_train_file, output_test_file)