import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
import streamlit as st
import json
import os  
import pandas as pd
from streamlit_option_menu import option_menu
from config.graphrag_prompt_emotion import search_prompt, final_prompt
from service.eval_mix_thread_emo import get_graphrag_answer_with_query


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "frontend" not in st.session_state:
        st.session_state.frontend = 'main'
    if "show_level" not in st.session_state:
        st.session_state.show_level = 1  # 1:Results only, 2:Results+Details, 3:Developer mode
    if "prompts" not in st.session_state:
        st.session_state.prompts = {
            "stage1": search_prompt.copy() if isinstance(search_prompt, dict) else {"system": "", "user": ""},
            "stage2": final_prompt.copy() if isinstance(final_prompt, dict) else {"system": "", "user": ""}
        }
    if "kb_state" not in st.session_state:
        st.session_state.kb_state = {
            "current_page": 1,  
            "text_content": [],  
            "kb_path": ""        
        }
    if "show_step1_table" not in st.session_state:
        st.session_state.show_step1_table = False

    if "config" not in st.session_state:
        st.session_state.config = {
            "test_output_path": "/vepfs/group04/user/qujx/MIC_demo/result_emotion_zh/output",
            "topN": 100,
            "threshold": 60
        }
    if "middle_results" not in st.session_state:
        st.session_state.middle_results = {}


def load_parquet_text_column(parquet_dir, target_file="create_final_text_units.parquet"):
    """Load text column content from Parquet file (return content list + error message)"""
    parquet_path = Path(parquet_dir) / target_file
    try:
        if not parquet_path.exists():
            return [], f"⚠️ Parquet file does not exist: {str(parquet_path)}"
        df = pd.read_parquet(parquet_path, columns=["text"])
        text_content = df["text"].dropna().astype(str).tolist()
        return text_content, None
    except Exception as e:
        error_msg = f"⚠️ Failed to load Parquet: {type(e).__name__} - {str(e)}"
        return [], error_msg


def clean_escape_chars(text):
    """Uniformly handle escape characters: remove quotes and slashes, convert \\n to actual newline"""
    if not isinstance(text, str):
        return text
    return text \
        .replace('\\"', '"') \
        .replace("\\'", "'") \
        .replace('\\\\', '\\') \
        .replace('\\n', '\n') \
        .replace('\\t', '\t')


def convert_ndarray_to_list(obj):
    """Recursively convert ndarray to list to solve JSON serialization issues"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    else:
        return obj


def main():
    st.set_page_config(page_title="GraphRAG Q&A System")   
    init_session_state()
    
    with st.sidebar:
        st.write(f'**GraphRAG Q&A System**')
        st.write(f'**Version: 1.0.0**')
        
        cols = st.columns(2)
        with cols[0]:
            reset_button = st.button("**�� Clear Chat**")
            if reset_button:
                st.session_state.messages = []
                st.success("Chat cleared")
        
        with cols[1]:
            regen_button = st.button("**�� Regenerate**")
        
        mode_selected = option_menu(None, ["Results Only", 'Results+Details', 'Developer Mode'],
                              icons=['chat-left-text', 'list-check', 'hammer'], 
                              menu_icon="cast", default_index=st.session_state.show_level - 1)
        st.session_state.show_level = 1 if mode_selected == "Results Only" else (2 if mode_selected == "Results+Details" else 3)

        
        with st.expander('**Save Options**'): 
            file_name = st.text_input('**Save File Name**', 'graphrag_results_{date}.json')
            save_button = st.button("**�� Save Results**")
            if save_button:
                if not file_name.strip():
                    st.error("Please enter a file name")
                else:
                    save_data = {
                        "messages": st.session_state.messages,
                        "prompts": st.session_state.prompts,
                        "config": st.session_state.config
                    }
                    with open(file_name, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, ensure_ascii=False, indent=2)
                    st.success(f"Results saved to: {os.path.abspath(file_name)}")

        
        with st.expander("**Configuration Parameters**", expanded=False):

            st.session_state.config["test_output_path"] = st.text_input(
                "Input Path",
                value=st.session_state.config["test_output_path"],
                placeholder="/path/to/output"
            )

            if not os.path.exists(st.session_state.config["test_output_path"]):
                st.warning(f"⚠️ Path does not exist: {st.session_state.config['test_output_path']}")
            
            st.session_state.config["topN"] = st.slider(
                "topK", 0, 100, st.session_state.config["topN"], 1
            )
            
            st.session_state.config["threshold"] = st.slider(
                "Threshold", 0, 100, st.session_state.config["threshold"], 5
            )

            if st.button("**�� View Knowledge Base**"):
                st.session_state.frontend = 'knowledge_base' 
                st.session_state.kb_state["kb_path"] = st.session_state.config["test_output_path"]  
                if not st.session_state.kb_state["text_content"]:
                    text_content, _ = load_parquet_text_column(st.session_state.config["test_output_path"])
                    st.session_state.kb_state["text_content"] = text_content

            if st.button("**�� View Community Report Table**"):
                st.session_state.frontend = 'step1_table' 
        
        
        st.markdown("#### Search Prompt")
        stage1_initial = json.dumps(
            st.session_state.prompts["stage1"], 
            ensure_ascii=False, 
            indent=2
        )
        stage1_full = st.text_area(
            "",
            height=350,
            value=stage1_initial,
            placeholder='{"system": "Your system prompt...", "user": "Your user prompt containing {data_table} and {user_input}..."}'
        )
        try:
            stage1_parsed = json.loads(stage1_full)
            if "system" in stage1_parsed and "user" in stage1_parsed:
                st.session_state.prompts["stage1"] = stage1_parsed
                if "{data_table}" not in stage1_parsed["user"] or "{user_input}" not in stage1_parsed["user"]:
                    st.warning("⚠️ User field must contain {data_table} and {user_input} placeholders")
                else:
                    st.success("✅ Format correct")
            else:
                st.error("❌ Missing 'system' or 'user' field")
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON format error: {str(e)}")
        
        st.markdown("#### Final Prompt")
        stage2_initial = json.dumps(
            st.session_state.prompts["stage2"], 
            ensure_ascii=False, 
            indent=2
        )

        stage2_full = st.text_area(
            "",
            height=350,
            value=stage2_initial,
            placeholder='{"system": "Your system prompt...", "user": "Your user prompt containing {report_data} and {user_input}..."}'
        )
        try:
            stage2_parsed = json.loads(stage2_full)
            if "system" in stage2_parsed and "user" in stage2_parsed:
                st.session_state.prompts["stage2"] = stage2_parsed
                if "{report_data}" not in stage2_parsed["user"] or "{user_input}" not in stage2_parsed["user"]:
                    st.warning("⚠️ User field must contain {report_data} and {user_input} placeholders")
                else:
                    st.success("✅ Format correct")
            else:
                st.error("❌ Missing 'system' or 'user' field")
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON format error: {str(e)}")
    
    if st.session_state.frontend == 'knowledge_base':
        st.subheader("�� GraphRAG Knowledge Base Viewer")
        st.markdown("<font color=gray size=2>**Note: Currently viewing path is the output address configured in the sidebar, supports pagination browsing and content copying**</font>", unsafe_allow_html=True)
        
        kb_path = st.session_state.kb_state["kb_path"]
        text_content = st.session_state.kb_state["text_content"]
        current_page = st.session_state.kb_state["current_page"]
        
        if not os.path.exists(kb_path):
            st.error(f"⚠️ Knowledge base path does not exist: {kb_path}")
            new_kb_path = st.text_input("Please re-enter knowledge base path", value=kb_path)
            if st.button("**Reload**"):
                if os.path.exists(new_kb_path):
                    text_content, load_error = load_parquet_text_column(new_kb_path)
                    if load_error:
                        st.error(load_error)
                    else:
                        st.session_state.kb_state["text_content"] = text_content
                        st.session_state.kb_state["kb_path"] = new_kb_path
                        st.session_state.kb_state["current_page"] = 1
                        st.rerun()  
                else:
                    st.warning("Path still does not exist, please check and try again")
            back_button = st.button("**Return to Chat Page**")
            if back_button:
                st.session_state.frontend = 'main'
            return
        
        if not text_content:
            with st.spinner("Loading knowledge base..."):
                text_content, load_error = load_parquet_text_column(kb_path)
                if load_error:
                    st.error(load_error)
                    back_button = st.button("**Return to Chat Page**")
                    if back_button:
                        st.session_state.frontend = 'main'
                    return
                st.session_state.kb_state["text_content"] = text_content
        
        page_size = 10
        total_items = len(text_content)
        total_pages = (total_items + page_size - 1) // page_size
        
        cols = st.columns(3)
        with cols[0]:
            st.write(f"**Total items: {total_items} | Total pages: {total_pages}**")
            current_page = st.number_input(
                "Current Page",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                key="kb_page_input"
            )
            st.session_state.kb_state["current_page"] = current_page  
        
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_items)
        current_page_data = text_content[start_idx:end_idx]
        
        st.text_area(
            f"Knowledge Base Content (Page {current_page}/{total_pages})",
            value="\n\n".join([
                f"[{start_idx + i + 1}] {content[:800]}{'...' if len(content) > 800 else ''}" 
                for i, content in enumerate(current_page_data)
            ]),
            height=600,
            disabled=False  
        )
        
        back_button = st.button("**Return to Chat Page**", type="primary")
        if back_button:
            st.session_state.frontend = 'main'
    

    elif st.session_state.frontend == 'step1_table':
        st.subheader("�� Community Report Table")
        
        if "middle_results" in st.session_state:
            middle = st.session_state.middle_results
            if "data" in middle and "reports_list" in middle["data"]:
                reports_list = middle["data"]["reports_list"]
                if reports_list:
                    display_fields = ["report_idx", "title", "summary", "rank", "rank_explanation", "findings", "size"]
                    
                    for report in reports_list:
                        report["summary"] = report.get("summary", "")[:100] + "..." if len(report.get("summary", "")) > 100 else report.get("summary", "")
                        report["findings"] = report.get("findings", "")[:100] + "..." if len(report.get("findings", "")) > 100 else report.get("findings", "")
                    
                    st.dataframe(
                        pd.DataFrame(reports_list)[display_fields],
                        use_container_width=True,
                        hide_index=True,
                        column_config={col: st.column_config.TextColumn(col, width="medium") for col in display_fields}
                    )
                else:
                    st.info("ℹ️ No first-step data table content retrieved")
            else:
                st.info("ℹ️ No first-step data table content retrieved")
        else:
            st.info("ℹ️ Please perform a query first to get data before viewing the first-step table")
        
        if st.button("**Return to Chat Page**", type="primary"):
            st.session_state.frontend = 'main'


    else:
        st.subheader("GraphRAG Q&A System")
        st.markdown("<font color=green size=3>**● Please configure parameters in the left sidebar, enter your question below and get the answer.**</font>",
                    unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        query = st.chat_input("Please enter your question...")
        chat_input_regen = ""

        if regen_button and st.session_state.messages:
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "user":
                    chat_input_regen = msg["content"]
                    break

            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                st.session_state.messages.pop()
        
        if query or chat_input_regen:
            chat_input = chat_input_regen if chat_input_regen else query
            with st.chat_message("user"):
                st.markdown(chat_input)
            st.session_state.messages.append({"role": "user", "content": chat_input})
            
            stage1 = st.session_state.prompts["stage1"]
            stage2 = st.session_state.prompts["stage2"]
            if not stage1["system"] or not stage1["user"]:
                with st.chat_message("assistant"):
                    st.error("❌ Stage 1 Prompt (Search Prompt) System and User fields cannot be empty")
                st.session_state.messages.append({"role": "assistant", "content": "❌ Stage 1 Prompt incomplete, please fill in System and User fields"})
                return
            if not stage2["system"] or not stage2["user"]:
                with st.chat_message("assistant"):
                    st.error("❌ Stage 2 Prompt (Final Prompt) System and User fields cannot be empty")
                st.session_state.messages.append({"role": "assistant", "content": "❌ Stage 2 Prompt incomplete, please fill in System and User fields"})
                return

            if not os.path.exists(st.session_state.config["test_output_path"]):
                with st.chat_message("assistant"):
                    st.error(f"❌ Path does not exist: {st.session_state.config['test_output_path']}, please check and try again")
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Path does not exist: {st.session_state.config['test_output_path']}"})
                return
            
            with st.chat_message("assistant"):
                with st.spinner("�� Processing... (May take longer for large datasets)"):
                    try:                    
                        result = get_graphrag_answer_with_query(
                                query=(chat_input_regen or query or "").strip(),
                                data_path=st.session_state.config["test_output_path"],
                                stage1_prompt=stage1,          
                                stage2_prompt=stage2,
                                threshold=st.session_state.config["threshold"], 
                                topN=st.session_state.config["topN"]
                            )
                        result = convert_ndarray_to_list(result)
                        st.session_state.middle_results = result.get("middle_results", {})
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        st.exception(e)
                        st.session_state.messages.append({"role": "assistant", "content": f"❌ Processing failed: {str(e)}"})
                        return
                
                
                if st.session_state.show_level == 2:
                    st.subheader("Complete Intermediate Results JSON")
                    st.json(result, expanded=False)
                
                if st.session_state.show_level == 3:
                    st.subheader("Intermediate Results Details")
                    middle = result.get("middle_results", {})
                    
                    st.subheader("1. Search Prompt")
                    search_msgs_list = middle.get("data", {}).get("search_messages_list", [])
                    search_results = middle.get("data", {}).get("search_results", [])

                    if search_msgs_list:
                        st.info(f"Total {len(search_msgs_list)} community reports, containing prompts and corresponding generated content:")
                        for idx, msg in enumerate(search_msgs_list, 1):
                            st.markdown(f"### Report {idx}")
                            
                            st.markdown("#### 1.1 Prompt")
                            try:
                                msg_json_str = json.dumps(msg, ensure_ascii=False)
                                msg_parsed = json.loads(msg_json_str)
                                msg_str = json.dumps(msg_parsed, ensure_ascii=False, indent=2)
                            except Exception as e:
                                msg_str = json.dumps(msg, ensure_ascii=False, indent=2)
                            cleaned_msg_str = clean_escape_chars(msg_str)
                            st.code(cleaned_msg_str, language="json", line_numbers=True)
                            
                            st.markdown("#### 1.2 Generated Content")
                            report_content = next(
                                (item.get("Instruction") for item in search_results if item.get("report_idx") == idx),
                                None
                            )
                            if report_content and isinstance(report_content, str):
                                cleaned_content = report_content.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\').replace('\"', '"').replace('\'', "'")
                                if len(cleaned_content) > 1000:
                                    with st.expander("View Full Content", expanded=False):
                                        st.code(cleaned_content, language="text", line_numbers=True)
                                else:
                                    st.code(cleaned_content, language="text", line_numbers=True)
                            else:
                                st.info("ℹ️ No generated content retrieved for this report")
                            
                            if idx != len(search_msgs_list):
                                st.markdown("---")
                    else:
                        st.info("ℹ️ Complete report prompts or generated content not transmitted by backend")
                    
                    st.subheader("2. Second Step Input")
                    second_prompt_data = middle.get("data", {}).get("prompts", {}).get("second_prompt", {})
                    report_data = second_prompt_data.get("report_data", "")
                    if report_data:
                        st.code(report_data, language="text", line_numbers=True)
                    else:
                        st.info("ℹ️ No second-step retrieval results organized content retrieved")
                    
                    st.subheader("3. Final Prompt")
                    formatted_final = second_prompt_data.get("formatted_prompt", {})
                    if formatted_final and isinstance(formatted_final, dict):
                        formatted_str = json.dumps(formatted_final, ensure_ascii=False, indent=2)
                        cleaned_formatted_str = clean_escape_chars(formatted_str)
                        st.code(cleaned_formatted_str, language="json", line_numbers=True)
                    else:
                        st.info("ℹ️ No replaced Final Prompt retrieved")
                  

                st.subheader("Final Answer")
                if result.get("status") == "success":
                    st.success("✅ Processing completed")
                    st.markdown(result["response"])
                    final_answer = result["response"]
                else:
                    st.warning(f"⚠️ Processing status: {result.get('status')}")
                    st.markdown(result["response"])
                    final_answer = f"Processing status: {result.get('status')}\n{result['response']}"
            
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

 

if __name__ == "__main__":
    main()