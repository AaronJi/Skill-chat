import json
import os
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import argparse
import datetime

import time
import hmac
import base64
import requests
# from config import *
from hashlib import sha256
import os
import re
from openai import OpenAI
import time
from graphrag_prompt import search_prompt,final_prompt,base_prompt, reverse_base_prompt, reverse_final_prompt,reverse_search_prompt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pandas as pd
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
)
LN = "\n"
URL_HOST=""


class GBOP():
    def __init__(self, path, model, ak, sk, api, host, full_function=True):
        self.method = 'POST'
        self.path = path  # '/v2/chat/completions'
        self.queries = {}  # {'access_token':'24.cc7e366d65aa0bf44f90dd7bc306766c.2592000.1738834255.282335-116915560'}
        self.model = model
        self.access_key = ak
        self.secret_key = sk
        self.api_key = api
        self.host = host
        self.full_function = full_function

    def __call__(self, http_body):
        url = self._build_url()
        headers = self._build_headers()

        headers["X-DashScope-SSE"] = "enable"
        pattern = r'"content":"([^"]+)"'
        # think_pattern
        for attempt in range(15):
            try:
                result = ''
                res = requests.post(url, headers=headers, data=json.dumps(http_body), stream=True)
                for response in res.iter_lines():
                    # print(response)
                    i = response.decode("UTF-8")
                    matches = re.findall(pattern, i)
                    if matches:
                        i2 = matches[0]
                        result += i2
                print(result)
                return result
            except:
                if attempt < 14:
                    time.sleep(10)
                    print(f"request fail， {attempt + 1} retry!!")

    def chat(self, http_body):
        url = self._build_url()
        headers = self._build_headers()
        # res = requests.post(url, json=http_body, headers=headers)
        pattern = r'"content":"([^"]+)"'

        for attempt in range(10):
            try:
                result = ''
                res = requests.post(url, headers=headers, data=json.dumps(http_body))
                result = eval(res.content)
                return result
            except requests.exceptions.RequestException as e:
                if attempt < 9:
                    time.sleep(30)

    def _build_url(self):
        query_string = self._build_query_string()
        format_string = "%s%s?%s" if query_string else "%s%s%s"
        url = "http://" + format_string % (self.host, self.path, query_string)
        return url

    def _build_query_string(self):
        query_string = ""
        if not self.queries:
            return query_string
        queries_dic = sorted(self.queries)
        for k in queries_dic:
            if type(self.queries[k]) == list:
                self.queries[k].sort()
                for i in self.queries[k]:
                    item_string = "%s=%s" % (k, i)
                    query_string = query_string + item_string + "&"
            else:
                item_string = "%s=%s" % (k, self.queries[k])
                query_string = query_string + item_string + "&"
        query_string = query_string.strip("&")
        return query_string

    def _build_headers(self):
        gmt_format = '%a, %d %b %Y %H:%M:%S GMT'

        headers = {
            "X-Gapi-Ca-Timestamp": str(int(round(time.time() * 1000))),
            "X-Gapi-Ca-Algorithm": "hmac-sha256",
            "X-Gapi-Ca-Access-Key": self.access_key,
            "X-Gapi-Ca-Signed-Headers": 'X-Gapi-Ca-Timestamp',
            "Date": datetime.datetime.utcnow().strftime(gmt_format),
            "Host": self.host,
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json"
        }
        headers["X-Gapi-Ca-Signature"] = self._generate_signature(headers)
        return headers


    def _generate_signature(self, headers):
        query_string = self._build_query_string()
        content_array = [self.method, self.path, query_string]
        if headers:
            if headers.__contains__("X-Gapi-Ca-Access-Key"):
                content_array.append(headers["X-Gapi-Ca-Access-Key"])
            if headers.__contains__("Date"):
                content_array.append(headers["Date"])
            if headers.__contains__("X-Gapi-Ca-Signed-Headers"):
                custom_headers = headers["X-Gapi-Ca-Signed-Headers"].split(";")
                for custom_header in custom_headers:
                    content_array.append(custom_header + ":" + headers[custom_header])
        content = LN.join(content_array) + LN
        content = content.encode("utf-8")
        secret_key = self.secret_key.encode("utf-8")
        signature = base64.b64encode(hmac.new(secret_key, content, digestmod=sha256).digest())
        return signature

class Graphrag(GBOP):
    def __init__(self, path,is_reverse = False,model="llama3.3-70b-instruct",full_function=False):
        super().__init__(
            '/api/v1/services/aigc/text-generation/generation',
            model,
            '0f5d9a4844455632333331',
            '9b2013cb-e72f-4716-b56a-7d156fd71f72',
            'sk-fcdbd19de51648f4bb1fa277169cd0d4',
            'qwen-max.api-dev.test.geely.svc',
            full_function
        )
        INPUT_DIR = f"{path}/output"
        COMMUNITY_TABLE = "create_final_communities"
        COMMUNITY_REPORT_TABLE = "create_final_community_reports"
        ENTITY_TABLE = "create_final_nodes"
        ENTITY_EMBEDDING_TABLE = "create_final_entities"
        COMMUNITY_LEVEL = 0
        community_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_TABLE}.parquet")
        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

        communities = read_indexer_communities(community_df, entity_df, report_df)
        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
        entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
        sorted_reports=sorted(reports, key=lambda x: x.size, reverse=True)
        self.all_reports = sorted_reports[:10]
        self.is_reverse = is_reverse


    def infer_formalization(self, messages, k=None, url=None):
        if url is None:
            url = URL_HOST
        # process community by morals
        final_report = []
        for report in self.all_reports:
            t='id|title|occurrence weight|content|rank\n1|'+report.title+'1|'+report.full_content+'|' + str(report.rank)
            final_report.append(t)
        chat_input=messages[0]['content']
        real_response_final={'Instructions':[]}
        for t in final_report: #graph_table_llama3_short
            if self.is_reverse:
                search_body = [
                {'role':'system','content':reverse_search_prompt['system']},
                {'role':'user','content':reverse_search_prompt['user'].format(data_table = t,
                                                                   user_input = chat_input)}]
            else:
                search_body = [
                    {'role':'system','content':search_prompt['system']},
                    {'role':'user','content':search_prompt['user'].format(data_table = t,
                                                                        user_input = chat_input)}]
            search_response=self.vllm_service(search_body, url)
            search_response = (
                search_response.replace("{{", "{")
                .replace("}}", "}")
                .replace('"[{', "[{")
                .replace('}]"', "}]")
                .replace("\\", "")
                .replace("\\n", "")
                .replace("\n", "")
                .replace("\r", "")
                .replace('{"{"Instructions"','{"Instructions"')
                .strip()
            )
            # breakpoint()
            # print('search_response:\n',search_response)
            try:
                pattern = r'"Instruction":\s*"([^"]+)"\s*,\s*"score":\s*(\d+)'

                matches = re.findall(pattern, search_response)

                real_response = {'Instructions':[{"Instruction": answer, "score": int(score)} for answer, score in matches]}
            except:
                real_response={'Instructions':[{'Instruction':'','score':-1}]}
            real_response_final['Instructions']+=real_response['Instructions']
        print(real_response_final)
        key_points=[]
        for index,element in enumerate(real_response_final['Instructions']):
            if not isinstance(element, dict):
                continue
            if "Instruction" not in element or "score" not in element:
                continue
            key_points.append({
                "analyst": index,
                "Instruction": element["Instruction"],
                "score": element["score"],
            })

        # breakpoint()
        filtered_key_points = [
            point
            for point in key_points
            if point["score"] > 80  # type: ignore
        ]
        filtered_key_points = sorted(
            filtered_key_points,
            key=lambda x: x["score"],  # type: ignore
            reverse=True,  # type: ignore
        )
        data = []
        for point in filtered_key_points[:k]:
            formatted_response_data = []
            formatted_response_data.append(
                f'----Instruction {point["analyst"] + 1}----'
            )
            formatted_response_data.append(
                f'Importance Score: {point["score"]}'  # type: ignore
            )
            formatted_response_data.append(point["Instruction"])  # type: ignore
            formatted_response_text = "\n".join(formatted_response_data)
            data.append(formatted_response_text)
        text_data = "\n\n".join(data)
        if self.is_reverse:
            answer_body = [
                    {'role':'system','content':reverse_final_prompt['system']},
                    {'role':'user','content':reverse_final_prompt['user'].format(report_data=text_data,
                                                                        user_input=chat_input)}]
        else:
            answer_body = [
                    {'role':'system','content':final_prompt['system']},
                    {'role':'user','content':final_prompt['user'].format(report_data=text_data,
                                                                        user_input=chat_input)}]
        #print(answer_body)
        return self.vllm_service(answer_body, url)

    def infer_baseline(self, messages, serve='vllm'):
        chat_input=messages[0]['content']
        if self.is_reverse:
            second_prompt = reverse_base_prompt['user'].format(user_input=chat_input)
            temp_message = [
                {
                    "role": "system",
                    "content": reverse_base_prompt['system']
                },
                {
                    "role": "user",
                    "content": second_prompt
                }
            ]
        else:
            second_prompt = base_prompt['user'].format(user_input=chat_input)
            temp_message = [
                {
                    "role": "system",
                    "content": base_prompt['system']
                },
                {
                    "role": "user",
                    "content": second_prompt
                }
            ]
        print(temp_message)
        if serve=='vllm':
            return self.vllm_service_baseline(temp_message)

    def vllm_service(self, messages, url):
        client = OpenAI(api_key='EMPTY', base_url=url)
        response = client.chat.completions.create(
            model="Llama-3.3-70B-Instruct",  # deepseek-reasoner  deepseek-chat
            messages=messages,
            stream=True
        )
        # BEHAVIORS,EMOTION STATE,SITUATION,TOPIC,HUMOR SKILLS,COMFORT TECHNIQUES,CONVERSATION SKILLS,PERSON,EXAMPLES
        result = ""
        for item in response:
            # print(item)
            if item.choices[0].delta.content:
                result += item.choices[0].delta.content
                # print(result)
        return result

    def vllm_service_baseline(self, messages):
        client = OpenAI(api_key='EMPTY', base_url='http://10.209.76.161:8501/v1')
        response = client.chat.completions.create(
            model="Llama-3.3-70B-Instruct",  # deepseek-reasoner  deepseek-chat
            messages=messages,
            stream=True
        )
        # BEHAVIORS,EMOTION STATE,SITUATION,TOPIC,HUMOR SKILLS,COMFORT TECHNIQUES,CONVERSATION SKILLS,PERSON,EXAMPLES
        result = ""
        for item in response:
            # print(item)
            if item.choices[0].delta.content:
                result += item.choices[0].delta.content
        return result


gh = Graphrag("/vepfs/DI/user/wyh/Personal_Chat_Agent/GYAFC_demo/rag_dir", False)  # Fasle Formal// True Informal
message = [{"role": "assistant",
            "content": "Certainly . Jackie Chan is a very famous movie star in Hong Kong . His Chinese name is Cheng Long ."}]
response = gh.infer_formalization(message)
print(message)
print(response)