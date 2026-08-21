import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
import concurrent.futures
import json
import os.path

import requests
import tqdm

from utils.customgraphrag_emotion import GraphragV2


def write_data(data, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'messages': data}, ensure_ascii=False) + '\n')


def post_interface(messages, version):
    url = 'http://10.209.76.111:8503/graphrag_result'
    headers = {'Content-Type': 'application/json'}
    data = {
        'message': messages,
        'version': version
    }
    response = requests.post(url, data=json.dumps(data), headers=headers)
    text = response.text
    return text


def get_graphrag_answer_with_query(query, data_path, stage1_prompt, stage2_prompt, threshold=50, topN=10000):
    def validate_prompt(prompt, prompt_name):
        if not isinstance(prompt, dict):
            raise TypeError(f"{prompt_name} must be a dictionary, actual type: {type(prompt)}")
        required_keys = ["system", "user"]
        missing_keys = [key for key in required_keys if key not in prompt]
        if missing_keys:
            raise ValueError(f"{prompt_name} missing required keys: {missing_keys}")
        if not all(isinstance(prompt[key], str) for key in required_keys):
            raise TypeError(f"{prompt_name} 'system' and 'user' must be strings")
    
    try:
        validate_prompt(stage1_prompt, "stage1_prompt")
        validate_prompt(stage2_prompt, "stage2_prompt")
        
        rag = GraphragV2(
            INPUT_DIR=data_path,
            search_stage1_prompt=stage1_prompt,  
            search_stage2_prompt=stage2_prompt
        )
        
        result = rag(messages=[{"role": "user", "content": query}], threshold=threshold, topN=topN)
        
        return {
            "response": result["final_answer"],
            "middle_results": result["middle_results"],
            "status": result["status"],
            "threshold": threshold,
            "topN": topN
        }
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        return {
            "response": error_msg,
            "status": "error",
            "error_detail": str(e),
            "prompt_validation": "failed" if "stage1_prompt" in str(e) or "stage2_prompt" in str(e) else "unknown"
        }


def process_content(content):
    id = content['id']
    prefix_len = len('Paraphrase the following tweet without any explanation before or after it:')
    input = content['input'][prefix_len:]
    data_path = '/vepfs/group04/user/chenteng/graphrag/lamp7_70b_time/' + id + '/output'

    result = get_graphrag_answer_with_query(
        query=input,
        data_path=data_path,
        stage1_prompt=search_prompt,
        stage2_prompt=final_prompt,
        threshold=80,
        topN=50
    )
    
    print("used_message".center(100, "="))
    if "middle_results" in result:
        print(json.dumps(result["middle_results"], ensure_ascii=False, indent=2))
    
    return {id: result["response"]}