import os
import json
import argparse
from openai import OpenAI


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sa")
    parser.add_argument("--domain", type=str, default="amazon")
    args = parser.parse_args()

    api_key = os.environ.get('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    output_file_path = f"./response/openai_batch/ood_nlp/{args.task}/{args.domain}/batch_results.json"
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8") as file:
            batch_dict = json.load(file)
    else: raise FileNotFoundError(f"File not found: {output_file_path}")

    c = 0
    for k in batch_dict.keys():
        batch_info = client.batches.retrieve(batch_dict[k]['batch_api_obj_id'])
        status = batch_info.status

        if status == 'completed':
            output_file_id = batch_info.output_file_id
            batch_dict[k]['output_file_id'] = output_file_id
            print(f"Batch {k} is completed : output file id: {output_file_id} Total: {batch_info.request_counts.total} Completed: {batch_info.request_counts.completed}")
        else:
            c += 1
            print(f"Batch {k} {status}: {round(batch_info.request_counts.completed/batch_info.request_counts.total*100, 3)}% [{batch_info.request_counts.completed} // {batch_info.request_counts.total}]")
    
    with open(output_file_path, 'w') as f:
        json.dump(batch_dict, f)

    if c == 0: print(f"All {args.task}||{args.domain} batches are completed")