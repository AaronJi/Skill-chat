import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
import os
import re
import json
import ast  
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
)

from config.graphrag_prompt_emotion import final_prompt, search_prompt, LLAMA_SERVICE_URL, LLAMA_SERVICE_URL_70B


class GraphragV2():
    def __init__(self, 
                INPUT_DIR="/vepfs/group04/user/qujx/MIC_demo/result_emotion/output",
                input_dir_list=[], 
                search_stage1_prompt=search_prompt,
                search_stage2_prompt=final_prompt,
                force_reports=None, 
                merge_community=True,
                model="Llama-3.3-70B-Instruct"):
                     
        print("="*50)
        print("--- [GraphRAG Initialization] Loading data files ---")
        if input_dir_list:
            current_input_dir = input_dir_list[0]  
        else:
            current_input_dir = INPUT_DIR  

        self.model = model
        self.search_prompt = search_stage1_prompt
        self.final_prompt = search_stage2_prompt
        self.reports = []  
        self.all_reports_obj = []  
        self.raw_data = {
            "community_reports": None,  
            "communities": None,        
            "nodes": None,              
            "entities": None,           
            "relationships": None,      
            "text_units": None
        }
        self.data = {"communities": []}
        self.text_units_map = {}
        self.doc_to_text_units = {}
        

        try:
            self.raw_data["community_reports"] = pd.read_parquet(os.path.join(current_input_dir, "create_final_community_reports.parquet"))
            self.raw_data["communities"] = pd.read_parquet(os.path.join(current_input_dir, "create_final_communities.parquet"))
            self.raw_data["nodes"] = pd.read_parquet(os.path.join(current_input_dir, "create_final_nodes.parquet"))
            self.raw_data["entities"] = pd.read_parquet(os.path.join(current_input_dir, "create_final_entities.parquet"))
            self.raw_data["relationships"] = pd.read_parquet(os.path.join(current_input_dir, "create_final_relationships.parquet"))
            self.raw_data["text_units"] = pd.read_parquet(os.path.join(current_input_dir, "create_final_text_units.parquet"))
            print(f"✅ Loaded text_units table: {len(self.raw_data['text_units'])} text units total")

            for _, tu_row in self.raw_data["text_units"].iterrows():
                tu_id = tu_row.get("id", "")
                tu_text = tu_row.get("text", "").strip()
                if tu_id and tu_text:
                    self.text_units_map[tu_id] = tu_text
                doc_ids_field = tu_row.get("document_ids", [])
    
                if isinstance(doc_ids_field, np.ndarray):
                    doc_ids = doc_ids_field.tolist()
                elif isinstance(doc_ids_field, str):
                    doc_ids = [d.strip() for d in doc_ids_field.strip("[]'\"").split(",") if d.strip()]
                else:
                    doc_ids = doc_ids_field if isinstance(doc_ids_field, list) else []
                
                for doc_id in doc_ids:
                    if doc_id not in self.doc_to_text_units:
                        self.doc_to_text_units[doc_id] = []
                    if tu_id not in self.doc_to_text_units[doc_id]:
                        self.doc_to_text_units[doc_id].append(tu_id)
            print(f"✅ text_units mapping completed: {len(self.text_units_map)} text units, {len(self.doc_to_text_units)} document associations")

            entity_id_map = {
                ent.get("id", ""): {
                    "title": ent.get("title", ""),
                    "type": ent.get("type", "unknown"),
                    "description": ent.get("description", "").strip()[:200]  
                } for _, ent in self.raw_data["entities"].iterrows() if ent.get("id")
            }
            rel_id_map = {
                rel.get("id", ""): {
                    "source": rel.get("source", ""),
                    "target": rel.get("target", ""),
                    "description": rel.get("description", "").strip()[:200]
                } for _, rel in self.raw_data["relationships"].iterrows() if rel.get("id")
            }

            def parse_id_list(id_field):
                if isinstance(id_field, np.ndarray):
                    return [str(id_val).strip() for id_val in id_field if pd.notna(id_val) and str(id_val).strip()]
                elif pd.isna(id_field) or id_field is None:
                    return []
                elif isinstance(id_field, str):
                    clean_str = id_field.strip().strip("[]'\"").replace("' ", ",").replace(" '", ",").replace(" ", "")
                    return [id_val.strip() for id_val in clean_str.split(",") if id_val.strip()]
                elif isinstance(id_field, list):
                    return [str(id_val).strip() for id_val in id_field if pd.notna(id_val) and str(id_val).strip()]
                else:
                    return []
            for _, comm_row in self.raw_data["communities"].iterrows():
                comm_entity_ids = parse_id_list(comm_row.get("entity_ids", ""))
                comm_rel_ids = parse_id_list(comm_row.get("relationship_ids", ""))
                comm_tu_ids = parse_id_list(comm_row.get("text_unit_ids", ""))

                community_info = {
                    "id": comm_row.get("id", ""),
                    "human_readable_id": int(comm_row.get("human_readable_id", -1)) if pd.notna(comm_row.get("human_readable_id")) else -1,
                    "title": comm_row.get("title", f"Community_{comm_row.get('human_readable_id', -1)}"),
                    "level": int(comm_row.get("level", 0)) if pd.notna(comm_row.get("level")) else 0,
                    "period": comm_row.get("period", "unknown time"),
                    "size": int(comm_row.get("size", 0)) if pd.notna(comm_row.get("size")) else 0,
                    "entities": [entity_id_map[ent_id] for ent_id in comm_entity_ids if ent_id in entity_id_map],
                    "relationships": [rel_id_map[rel_id] for rel_id in comm_rel_ids if rel_id in rel_id_map],
                    "text_units": [
                        {"tu_id": tu_id, "text": self.text_units_map[tu_id]} 
                        for tu_id in comm_tu_ids if tu_id in self.text_units_map
                    ],
                    "related_docs": list({
                        doc_id for tu_id in comm_tu_ids 
                        for doc_id in self.doc_to_text_units.get(tu_id, [])
                    })
                }
                self.data["communities"].append(community_info)
            print(f"✅ Community structure built: {len(self.data['communities'])} communities, each with complete associated information")

            comm_human_to_info = {
                comm["human_readable_id"]: comm 
                for comm in self.data["communities"] if comm["human_readable_id"] != -1
            }
            for idx, report_row in self.raw_data["community_reports"].iterrows():
                comm_human_id = int(report_row.get("community", -1))
                comm_info = comm_human_to_info.get(comm_human_id, {})
                
                report_obj = type('Report', (object,), {
                    "report_id": report_row.get("id", ""),   
                    "community": comm_info.get("title", f"Community_{comm_human_id}"),
                    "community_id": comm_info.get("id", ""), 
                    "community_title": report_row.get("title", ""),
                    "summary": report_row.get("summary", ""), 
                    "report_title": report_row.get("title", ""),
                    "rank": report_row.get("rank", 0),
                    "rank_explanation": report_row.get("rank_explanation", 0),
                    "findings": report_row.get("findings", 0),
                    "related_text_units": comm_info.get("text_units", []),  
                    "related_entities": comm_info.get("entities", []),
                    "human_readable_id": comm_human_id
                })
                self.all_reports_obj.append(report_obj)

                self.reports.append(
                    f"{report_obj.report_id}|{report_obj.report_title}|1|{report_row.get('full_content', '')[:5000]}|{str(report_row.get('rank', 0))}"
                )

                self.reports_list = []
                for idx, report_obj in enumerate(self.all_reports_obj):
                    self.reports_list.append({
                        "report_idx": idx + 1,
                        "title": report_obj.report_title,
                        "summary": report_obj.summary,
                        "rank": report_obj.rank,
                        "rank_explanation": report_obj.rank_explanation,
                        "findings": report_obj.findings,
                        "size": len(report_obj.related_text_units) if hasattr(report_obj, "related_text_units") else 0
                    })

            print(f"\n✅ Initialization completed - data statistics:")
            print(f"   - Total community reports: {len(self.all_reports_obj)}")
            print(f"   - Associated documents: {len(self.text_units_map)}")
            print(f"   - Total communities: {len(self.raw_data['communities'])}")
            print(f"   - Total nodes (community-entity associations): {len(self.raw_data['nodes'])}")
            print(f"   - Total entities: {len(self.raw_data['entities'])}")
            print(f"   - Total relationships: {len(self.raw_data['relationships'])}")

        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            import traceback
            traceback.print_exc()  

        print("="*50 + "\n")


    def vllm_service(self, messages, url=LLAMA_SERVICE_URL_70B, model_name="Llama-3.3-70B-Instruct", temperature=0.0):
        print(f"\n�� [LLM Call] Model: {model_name} | Temperature: {temperature}")
        
        normalized_messages = []
        try:
            if not isinstance(messages, list):
                messages = [messages]
            
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")  
                    content = str(msg.get("content", ""))  
                    normalized_messages.append({"role": role, "content": content})
                else:
                    normalized_messages.append({"role": "user", "content": str(msg)})
        
        except Exception as e:
            print(f"   Message format normalization failed: {str(e)}")
            normalized_messages = [{"role": "user", "content": "Failed to get message"}]
        
        try:
            if normalized_messages:
                first_content = normalized_messages[0]["content"]
                preview = first_content[:100] if len(first_content) > 100 else first_content
                print(f"   Standardized prompt first 100 characters: {preview}...")
        except Exception as e:
            print(f"   Prompt preview failed: {str(e)}")
        
        try:
            client = OpenAI(api_key="EMPTY", base_url=url)
            response = client.chat.completions.create(
                model=model_name,
                messages=normalized_messages,  
                max_tokens=1024,
                temperature=temperature,
                stream=True
            )
            result = ""
            for item in response:
                if item.choices[0].delta.content:
                    result += item.choices[0].delta.content
            
            result_preview = result[:100] if len(result) > 100 else result
            print(f"✅ [LLM Call] Completed! Return result: {result}")
            return result.strip()
        
        except Exception as e:
            err_msg = f"❌ [LLM Call Failed] {str(e)}"
            print(err_msg)
            return err_msg
    


    def _parse_community_entity_ids(self, entity_ids_str):
        try:
            formatted_str = entity_ids_str.replace("' '", "','").replace(" ", ",")
            return ast.literal_eval(formatted_str)
        except Exception as e:
            print(f"⚠️ [Parsing Warning] entity_ids parsing failed: {entity_ids_str} | Error: {e}")
            return []


    def _find_relevant_graph_info(self, relevant_report_indices):
        import numpy as np
        print("\n" + "-"*30)
        print("�� [Graph Information Extraction] Starting association of entities, relationships, communities")
        print(f"   Report indices to process: {relevant_report_indices}")

        def numpy_to_python(value):
            if isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (list, str, int, float, bool)):
                return value
            else:
                return str(value)

        comm_human_ids = set()
        relevant_communities = []
        for report_idx in relevant_report_indices:
            if 0 <= report_idx - 1 < len(self.all_reports_obj):
                report_obj = self.all_reports_obj[report_idx - 1]
                comm_human = report_obj.human_readable_id
                comm_uuid = report_obj.community_id
                comm_title = report_obj.community_title

                if comm_human not in comm_human_ids and comm_human != -1:
                    comm_human_ids.add(comm_human)
                    comm_mask = self.raw_data["communities"]["human_readable_id"] == comm_human
                    if comm_mask.any():
                        comm_row = self.raw_data["communities"][comm_mask].iloc[0]
                        relevant_communities.append({
                            "id": numpy_to_python(comm_uuid),
                            "human_readable_id": numpy_to_python(comm_human),
                            "title": numpy_to_python(comm_row.get("title", "")),
                            "level": numpy_to_python(comm_row.get("level", 0)),
                            "summary": numpy_to_python(comm_row.get("summary", "")),
                            "rank": numpy_to_python(comm_row.get("rank", 0))
                        })
                    else:
                        relevant_communities.append({
                            "id": numpy_to_python(comm_uuid),
                            "human_readable_id": numpy_to_python(comm_human),
                            "title": numpy_to_python(comm_title),
                            "level": 0,
                            "summary": numpy_to_python(report_obj.summary),
                            "rank": numpy_to_python(report_obj.rank)
                        })

        print(f"   Associated community human_readable_id: {sorted(comm_human_ids) if comm_human_ids else 'none'}")
        if relevant_communities:
            print(f"   Associated community details (first 1): {relevant_communities[0]['title']} (UUID: {relevant_communities[0]['id']})")
        else:
            print(f"   Associated community details: none (no community data matched)")

        node_mask = self.raw_data["nodes"]["community"].isin(comm_human_ids)
        relevant_node_rows = self.raw_data["nodes"][node_mask]
        entity_uuids = set(relevant_node_rows["id"].tolist())
        print(f"   Entity UUIDs associated with communities: {len(entity_uuids)}")

        entity_uuid_to_name = {}
        relevant_entities = []
        ent_mask = self.raw_data["entities"]["id"].isin(entity_uuids)
        for _, ent_row in self.raw_data["entities"][ent_mask].iterrows():
            ent_uuid = numpy_to_python(ent_row["id"])
            ent_name = numpy_to_python(ent_row["title"])
            entity_uuid_to_name[ent_uuid] = ent_name
            relevant_entities.append({
                "id": ent_uuid,
                "name": ent_name,
                "type": numpy_to_python(ent_row.get("type", "")),
                "description": numpy_to_python(ent_row.get("description", "")),
                "text_unit_ids": numpy_to_python(ent_row.get("text_unit_ids", []))
            })

        print(f"   Successfully associated entities: {len(relevant_entities)}")
        print(f"   First 3 entities: {[ent['name'] for ent in relevant_entities[:3]]}")

        relevant_relationships = []
        if len(relevant_entities) > 0:
            entity_names = set(entity_uuid_to_name.values())
            rel_mask = (self.raw_data["relationships"]["source"].isin(entity_names)) | \
                    (self.raw_data["relationships"]["target"].isin(entity_names))
            relevant_rel_rows = self.raw_data["relationships"][rel_mask]

            for _, rel_row in relevant_rel_rows.iterrows():
                relevant_relationships.append({
                    "id": numpy_to_python(rel_row["id"]),
                    "human_readable_id": numpy_to_python(rel_row.get("human_readable_id", 0)),
                    "source_entity": numpy_to_python(rel_row["source"]),  
                    "target_entity": numpy_to_python(rel_row["target"]),  
                    "description": numpy_to_python(rel_row.get("description", "")),
                    "weight": numpy_to_python(rel_row.get("weight", 0)),
                    "combined_degree": numpy_to_python(rel_row.get("combined_degree", 0)),
                    "text_unit_ids": numpy_to_python(rel_row.get("text_unit_ids", []))
                })

        print(f"   Successfully associated relationships: {len(relevant_relationships)}")
        if len(relevant_relationships) > 0:
            print(f"   First relationship: {relevant_relationships[0]['source_entity']} → {relevant_relationships[0]['target_entity']}")

        print("✅ [Graph Information Extraction] Completed!")
        print(f"   - Associated communities: {len(relevant_communities)}")
        print(f"   - Associated entities: {len(relevant_entities)}")
        print(f"   - Associated relationships: {len(relevant_relationships)}")
        print("-"*30)

        return {
            "communities": relevant_communities,
            "entities": relevant_entities,
            "relationships": relevant_relationships
        }



    def search(self, messages: List[Dict[str, str]], threshold=50, topN=10000, url=LLAMA_SERVICE_URL_70B) -> Dict[str, any]:
        standard_graphrag_result = {
            "status": "error",  
            "query": "",       
            "search_metadata": {
                "retrieval_time": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_reports_count": len(self.reports) if hasattr(self, "reports") else 0,
                "threshold": threshold,
                "topN": topN,
                "llm_service_url": url,
                "raw_results_count": 0,
                "filtered_results_count": 0,
                "graph_info_status": "not_fetched"
            },
            "data": {
                "communities": [],
                "entities": [],
                "relationships": [],
                "search_results": [],  
                "reports_list": self.reports_list,
                "prompts": {           
                    "first_prompt": {},
                    "second_prompt": {}
                },
                "search_messages_list": []
            }
        }

        try:
            print(f"Original messages: {messages}")
            user_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "user"]
            print(f"User messages: {user_messages}")

            if not user_messages:
                raise ValueError("No user query extracted from messages (needs message with role='user')")
            
            chat_input = user_messages[-1]["content"].strip()
            print(f"User input: '{chat_input}'")
            
            if not chat_input:
                raise ValueError("User query content cannot be empty")
            
            standard_graphrag_result["query"] = chat_input
            standard_graphrag_result["search_metadata"]["user_query"] = chat_input

            print("\n" + "="*50)
            print("�� [Search Phase] Starting to process user query")
            print(f"User question: {chat_input}")
            print(f"Search configuration: total reports={len(self.reports)} | threshold={threshold} | TopN={topN}")
            print("="*50)

            real_response_final = {"Instructions": []} 

            if not hasattr(self, "reports") or len(self.reports) == 0:
                raise RuntimeError("No report data loaded (self.reports is empty)")
            
            for idx, report in enumerate(self.reports):
                print(f"\n�� [Searching report {idx+1}/{len(self.reports)}]")
                if not hasattr(self, "search_prompt") or "user" not in self.search_prompt or "system" not in self.search_prompt:
                    raise RuntimeError("search_prompt format error, must contain 'system' and 'user' fields")
                
                user_prompt = self.search_prompt["user"].replace("{data_table}", report)
                user_prompt = user_prompt.replace("{user_input}", chat_input)
                
                search_messages = [
                    {"role": "system", "content": self.search_prompt["system"]},
                    {"role": "user", "content": user_prompt}
                ]
                
                print("vllm search prompt".center(100, "="))
                print(json.dumps(search_messages, ensure_ascii=False, indent=2))
                
                standard_graphrag_result["data"]["search_messages_list"].append(search_messages)

                if not hasattr(self, "vllm_service"):
                    raise AttributeError("'GraphragV2' object has no attribute 'vllm_service' (need to initialize LLM service first)")
                
                search_response = self.vllm_service(
                    messages=search_messages,
                    url=url,
                    temperature=0.0
                )

                search_response_clean = (
                    search_response.replace("{{", "{")
                    .replace("}}", "}")
                    .replace('"[{', "[{")
                    .replace('}]"', "}]")
                    .replace("\\", "")
                    .replace("\\n", " ")
                    .replace("\n", " ")
                    .replace("\r", " ")
                    .replace('{"{"Instructions"', '{"Instructions"')
                    .strip()
                )

                try:
                    pattern = r'"Instruction":\s*"([^"]+)"\s*,\s*"score":\s*(\d+)'
                    matches = re.findall(pattern, search_response_clean)
                    for instr, score in matches:
                        score_int = int(score)
                        print(f"   Extracted key information: Score={score_int} | First 50 characters: {instr[:50]}...")
                        real_response_final["Instructions"].append({
                            "report_idx": idx + 1,
                            "Instruction": instr,
                            "score": score_int
                        })
                except Exception as e:
                    err_msg = f"⚠️ [Parsing failed] Report {idx+1}: {str(e)}"
                    print(err_msg)
                    real_response_final["Instructions"].append({
                        "report_idx": idx + 1,
                        "Instruction": err_msg,
                        "score": -1
                    })

            raw_results_count = len(real_response_final["Instructions"])
            filtered = [item for item in real_response_final["Instructions"] if item["score"] > threshold]
            sorted_filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)[:topN]
            
            standard_graphrag_result["search_metadata"]["raw_results_count"] = raw_results_count
            standard_graphrag_result["search_metadata"]["filtered_results_count"] = len(sorted_filtered)
            standard_graphrag_result["data"]["search_results"] = sorted_filtered  
            standard_graphrag_result["data"]["prompts"]["first_prompt"] = self.search_prompt 

            print(f"\n�� [Result Filtering] Raw search results total: {raw_results_count}")
            print(f"   Filtered results: {len(sorted_filtered)} items (Top{topN}, threshold {threshold})")

            relevant_graph_info = {"communities": [], "entities": [], "relationships": []}
            if sorted_filtered:
                try:
                    unique_report_indices = list({item["report_idx"] for item in sorted_filtered})
                    relevant_graph_info = self._find_relevant_graph_info(unique_report_indices)
                    
                    standard_graphrag_result["data"]["communities"] = relevant_graph_info.get("communities", [])
                    standard_graphrag_result["data"]["entities"] = relevant_graph_info.get("entities", [])
                    standard_graphrag_result["data"]["relationships"] = relevant_graph_info.get("relationships", [])
                    standard_graphrag_result["search_metadata"]["graph_info_status"] = "success"

                    print("\n" + "="*30)
                    print("�� [Intermediate Results Display] Entity, relationship, community details")
                    print(f"Community information:\n{json.dumps(relevant_graph_info['communities'], indent=2, ensure_ascii=False)}")
                    print(f"\nEntity information:\n{json.dumps(relevant_graph_info['entities'], indent=2, ensure_ascii=False)}")
                    print(f"\nRelationship information:\n{json.dumps(relevant_graph_info['relationships'], indent=2, ensure_ascii=False)}")
                    print("="*30 + "\n")
                except Exception as e:
                    warn_msg = f"⚠️ [Graph association failed]: {str(e)}"
                    print(warn_msg)
                    standard_graphrag_result["search_metadata"]["graph_info_status"] = f"failed: {warn_msg}"
            else:
                print("\n⚠️ [No valid results] No search results meeting threshold found")
                standard_graphrag_result["search_metadata"]["graph_info_status"] = "no_valid_results"

            if not hasattr(self, "final_prompt") or "user" not in self.final_prompt or "system" not in self.final_prompt:
                raise RuntimeError("final_prompt format error, must contain 'system' and 'user' fields")
            
            if not sorted_filtered:
                text_data = "⚠️ No key information meeting threshold found"
            else:
                data = []
                for idx, point in enumerate(sorted_filtered[:topN], 1):
                    formatted_data = [
                        f'----Instruction {idx}----',
                        f'Importance Score: {point["score"]}',
                        point["Instruction"]
                    ]
                    data.append("\n".join(formatted_data))
                text_data = "\n\n".join(data)
            
            second_prompt_user = self.final_prompt["user"].format(
                report_data=text_data,
                user_input=chat_input
            )
            second_prompt = {
                "system": self.final_prompt.get("system", ""),  
                "user": second_prompt_user  
            }

            standard_graphrag_result["data"]["prompts"]["second_prompt"] = {
                "raw_template": self.final_prompt,
                "formatted_prompt": second_prompt,
                "report_data": text_data
            }

            print(f"\n�� [Second Phase Prompt] System prompt: {self.final_prompt['system']}")
            print(f"�� [Second Phase Prompt] User prompt: {second_prompt}")

            standard_graphrag_result["status"] = "success"
            standard_graphrag_result["search_metadata"]["message"] = "LLM search + graph association completed"
            return standard_graphrag_result

        except Exception as e:
            error_msg = str(e)
            standard_graphrag_result["status"] = "error"
            standard_graphrag_result["search_metadata"]["message"] = f"Search failed: {error_msg}"
            import traceback
            traceback.print_exc()
            return standard_graphrag_result

    def get_middle_results(self, query, search_results):
        middle_results = {
            "data": {
                "communities": [],
                "entities": [],
                "relationships": [],
                "search_results": search_results,
                "prompts": {
                    "first_prompt": self.search_prompt,
                    "second_prompt": self.final_prompt
                },
            },
            "debug": {  
                "retrieved_doc_ids": [],
                "community_docs_mapping": [],
                "matched_community_ids": [],
                "entity_ids_from_communities": [],
                "relationship_ids_from_communities": []
            }
        }

        try:
            retrieved_doc_ids = []
            for doc in search_results:
                if not isinstance(doc, dict):
                    continue
                doc_id = doc.get("id", "").strip()
                if doc_id:
                    retrieved_doc_ids.append(doc_id)
            middle_results["debug"]["retrieved_doc_ids"] = retrieved_doc_ids

            if not retrieved_doc_ids:
                middle_results["data"]["message"] = "No valid document IDs extracted from search results"
                return middle_results

            all_communities = self.data.get("communities", [])
            if not all_communities:
                middle_results["data"]["message"] = "Community data not loaded (self.data['communities'] is empty)"
                return middle_results

            doc_to_communities = {}
            for comm in all_communities:
                comm_id = comm.get("id", "")
                comm_human_id = comm.get("human_readable_id", -1)
                comm_docs = comm.get("related_docs", [])
                if isinstance(comm_docs, str):
                    comm_docs = [d.strip("'[] ") for d in comm_docs.split(",") if d.strip("'[] ")]
                elif not isinstance(comm_docs, list):
                    comm_docs = []

                middle_results["debug"]["community_docs_mapping"].append({
                    "community_id": comm_id,
                    "human_readable_id": comm_human_id,
                    "related_docs": comm_docs
                })

                for doc_id in comm_docs:
                    if doc_id not in doc_to_communities:
                        doc_to_communities[doc_id] = []
                    doc_to_communities[doc_id].append(comm)

            related_communities = []
            matched_comm_ids = set()
            for doc_id in retrieved_doc_ids:
                communities_for_doc = doc_to_communities.get(doc_id, [])
                for comm in communities_for_doc:
                    comm_id = comm.get("id", "")
                    if comm_id not in matched_comm_ids:
                        matched_comm_ids.add(comm_id)
                        related_communities.append(comm)
            middle_results["debug"]["matched_community_ids"] = list(matched_comm_ids)

            if not related_communities:
                middle_results["data"]["message"] = "No communities associated with retrieved documents (document IDs not in community related_docs)"
                return middle_results

            entity_ids = set()
            for comm in related_communities:
                for ent in comm.get("entities", []):
                    ent_id = ent.get("id") if isinstance(ent, dict) else ent
                    if ent_id:
                        entity_ids.add(str(ent_id))
            middle_results["debug"]["entity_ids_from_communities"] = list(entity_ids)

            related_entities = []
            if entity_ids:
                for ent_id in entity_ids:
                    ent_mask = self.raw_data["entities"]["id"].astype(str) == ent_id
                    if ent_mask.any():
                        ent_row = self.raw_data["entities"][ent_mask].iloc[0]
                        related_entities.append({
                            "id": ent_id,
                            "title": ent_row.get("title", "unknown entity"),
                            "type": ent_row.get("type", "unknown type"),
                            "description": ent_row.get("description", "")[:300]
                        })

            relationship_ids = set()
            for comm in related_communities:
                for rel in comm.get("relationships", []):
                    rel_id = rel.get("id") if isinstance(rel, dict) else rel
                    if rel_id:
                        relationship_ids.add(str(rel_id))
            middle_results["debug"]["relationship_ids_from_communities"] = list(relationship_ids)

            related_relationships = []
            if relationship_ids:
                for rel_id in relationship_ids:
                    rel_mask = self.raw_data["relationships"]["id"].astype(str) == rel_id
                    if rel_mask.any():
                        rel_row = self.raw_data["relationships"][rel_mask].iloc[0]
                        related_relationships.append({
                            "id": rel_id,
                            "source": rel_row.get("source", "unknown source"),
                            "target": rel_row.get("target", "unknown target"),
                            "description": rel_row.get("description", "")[:300]
                        })

            middle_results["data"]["communities"] = related_communities
            middle_results["data"]["entities"] = related_entities
            middle_results["data"]["relationships"] = related_relationships
            middle_results["data"]["message"] = (
                f"Successfully associated {len(related_communities)} communities, "
                f"{len(related_entities)} entities, {len(related_relationships)} relationships"
            )

        except Exception as e:
            middle_results["data"]["message"] = f"Association failed: {str(e)}"
            import traceback
            middle_results["debug"]["error_stack"] = traceback.format_exc()[:500]

        return middle_results
    


    def get_final_answer(self, messages, second_prompt, serve='vllm', url=LLAMA_SERVICE_URL_70B, model_name="Llama-3.3-70B-Instruct"):
        if messages[0]['role'] == 'system':
            messages[0]['content'] = second_prompt
        else:
            messages.insert(0, {"role": "system", "content": second_prompt})

        return self.vllm_service(messages, url, model_name=model_name)


    def __call__(self, messages, threshold=50, topN=10000, url=LLAMA_SERVICE_URL_70B, model_name="Llama-3.3-70B-Instruct"):
        search_output = self.search(
            messages=messages,
            threshold=threshold,
            topN=topN,
            url=url
        )

        second_prompt = {}
        if isinstance(search_output.get("data"), dict):
            prompts = search_output["data"].get("prompts", {})
            second_prompt = prompts.get("second_prompt", {})
            if not second_prompt:
                user_query = search_output.get("query", "User query not obtained")
                second_prompt = {
                    "system": "Based on provided information, concisely and accurately answer user questions",
                    "user": f"User question: {user_query}\nAvailable information: No eligible search results found, please inform user cannot answer"
                }

        final_answer = self.get_final_answer(
            messages=messages,
            second_prompt=second_prompt,  
            url=url,
            model_name=model_name
        )

        return {
            "search_output": search_output,
            "middle_results": search_output,
            "final_answer": final_answer,
            "status": search_output.get("status", "error")
        }



if __name__ == "__main__":
    rag = GraphragV2(
        INPUT_DIR="/vepfs/group04/user/qujx/MIC_demo/result_emotion/output"
    )
    
    test_query = "I'm almost out of my salary this month, and I'm wondering whether to use the remaining money for food or save it for my car loan."
    result = rag(
        messages=[{"role": "user", "content": test_query}],
        threshold=50,
        topN=3
    )
    
    print("\nFinal answer:")
    print(result["final_answer"])