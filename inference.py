import json
import tqdm
import concurrent.futures
from evaluate import Graphrag
import json
import requests
import re, os
import tqdm
import argparse


def write_data(data, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'messages': data}, ensure_ascii=False) + '\n')


def process_content(content, topk, url):
    data = json.loads(content)
    messages = data['messages']
    new_message = messages.copy()
    response = graphrag.infer_formalization(messages, topk, url)
    new_message[1]['content'] = response
    return new_message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get GYAFC response to JSON files.')
    parser.add_argument('--topk', type=int, default=None, help='topk for instructions. None means full')
    parser.add_argument('--url', type=str, default="http://10.209.76.196:8501/v1", help='url for llama3-70b')
    parser.add_argument('--reverse', type=bool, default=False, help='False: Informal->Formal; True:Formal->Informal')
    args = parser.parse_args()
    if args.reverse:
        output_dir = "/vepfs/DI/user/wyh/Personal_Chat_Agent/GYAFC_demo/result_0319/reverse"
    else:
        output_dir = "/vepfs/DI/user/wyh/Personal_Chat_Agent/GYAFC_demo/result_0319"
    graphpath = "/vepfs/DI/user/wyh/Personal_Chat_Agent/GYAFC_demo/rag_dir"
    querypath = "/vepfs/DI/user/wyh/Personal_Chat_Agent/GYAFC_demo/datasets/reverse_gyafc_test.json"
    if args.topk == None:
        outputpath = f"{output_dir}/Full_gyafc.json"
    else:
        outputpath = f"{output_dir}/Top{args.topk}_gyafc.json"
    graphrag = Graphrag(path=graphpath, is_reverse=args.reverse)

    with open(querypath, 'r', encoding='utf-8') as f:
        contents = f.readlines()
    # contents = contents[:5]

    WORKERS = 12

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(process_content, content, args.topk, args.url) for content in contents]

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures),
                                total=len(contents),
                                desc="Processing conversations"):
            try:
                result = future.result()
                # print("&&&&&",result)
                os.makedirs(os.path.dirname(outputpath), exist_ok=True)
                write_data(result, outputpath)
            except Exception as e:
                print(f"Error processing item: {e}")
