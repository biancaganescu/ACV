import json
import re
import sys
sys.path.append('../helper/')
from utils import print_api_usage, openai_chat_completion_with_retry, SurgeryKitUnitTest, SurgeryKitModule, setup_prompt_environment, load_json_answer
from models.OpenAI import OpenAIModel, APIUsageTracker, Validator
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pathlib

prompt_env= setup_prompt_environment("../prompt_templates")

def extract_context_from_trajectory(trajectory, context_list, engine):

    prompt_template = prompt_env.get_template("extract_context.j2")
    system_prompt = prompt_template.render(prompt_type="system")
    user_prompt = prompt_template.render(
        prompt_type="user",
        trajectory=trajectory,
        context_list=context_list
    )
    response = openai_chat_completion_with_retry(
        engine=engine,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        # max_tokens=1000,
        temperature=0.0
    )
    answer = response.choices[0].message['content'].strip()
    try:
        context = load_json_answer(answer)
        return context
    except Exception as e:
        print(f"Error parsing context from trajectory: {e}")
        return None


def parse_trajectory_to_dict(trajectory_text):
    """
    解析 trajectory，返回一个以 ID 为 Key 的字典。
    结构: { "doc_xxx": {data}, "msg_xxx": {data} }
    """
    lookup_table = {}
    
    def _extract_ids_recursive(obj):
            if isinstance(obj, dict):
                # 检查当前字典是否有 'id' 键
                if "id" in obj and isinstance(obj["id"], (str, int)):
                    # 存入查找表 (注意：如果 ID 重复，后出现的会覆盖先出现的)
                    # 将 id 转为字符串以保证 key 类型统一
                    obj_id = str(obj["id"])
                    lookup_table[obj_id] = obj
                
                # 继续遍历字典的所有值
                for value in obj.values():
                    _extract_ids_recursive(value)
                    
            elif isinstance(obj, list):
                # 如果是列表，遍历列表中的每个元素
                for item in obj:
                    _extract_ids_recursive(item)
    
    # 正则提取 Observation 后的 JSON 块
    pattern = r"Observation:\s*(\{.*?\})\s*(?=Action:|$)"
    matches = re.findall(pattern, trajectory_text, re.DOTALL)
    
    for json_str in matches:
        try:
            data = json.loads(json_str)
            _extract_ids_recursive(data)
                        
        except json.JSONDecodeError:
            print("Warning: Found invalid JSON block")
            continue
            
    return lookup_table



def identify_usage_of_entry(
    type,
    outline,
    action,
    entry,
    data,
    model,
    model_name
):
    prompt_template = prompt_env.get_template(f"identify_usage_{type}.j2")
    system_prompt = prompt_template.render(prompt_type="system")
    user_prompt = prompt_template.render(
        prompt_type="user",
        outline=outline,
        action=action,
        data=data,
        entry=entry
    )
    response = model.get_response_with_retry(
        model=model_name,
        input=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
    )
    answer = response.choices[0].message['content'].strip().lower()
    result = load_json_answer(answer)
    judgement = result['answer'].strip().lower()
    reasoning = str(result['reasoning']).strip()
    if judgement == 'yes':
        return True, reasoning
    elif judgement == 'no':
        return False, reasoning
    else:
        raise ValueError("Unrecognized answer when identifying usage of entry.")

def measure_usage_rate(
    type,
    outline,
    action,
    entry_list,
    model,
    model_name
):
    usage_cnt = 0
    retrieval_cnt = 0
    total_cnt = 0
    usage = []
    reasoning_list = []
    for entry, data in entry_list:
        total_cnt += 1
        if entry is not None:
            retrieval_cnt += 1
            is_used, reasoning = identify_usage_of_entry(type, outline, action, entry, data, model, model_name)
            if is_used:
                usage_cnt += 1
                usage.append(True)
            else:
                usage.append(False)
            reasoning_list.append(reasoning)
        else:
            usage.append(None)
            reasoning_list.append(None)
    # usage_rate_retrieved, usage_rate_total, retrieval_rate, usage, reasoning_list
    return usage_cnt / retrieval_cnt if retrieval_cnt else None, usage_cnt / total_cnt if total_cnt else None, retrieval_cnt / total_cnt if total_cnt else None, usage, reasoning_list

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data", help='Path to the input seed jsonl file')
    parser.add_argument('--agent_model_name', type=str, default="azure-grok-3", help="The model name used for the agent.")
    parser.add_argument('--agent_type', type=str, default="naive", help="The agent type used in the trajectory.")
    parser.add_argument('--essential_count', type=int, default=4, help='Number of essential data points to enrich')
    parser.add_argument('--sensitive_count', type=int, default=4, help='Number of sensitive data points to enrich')
    
    parser.add_argument('--engine', type=str, default='gpt-5.1-20251113', help='The language model engine to use for completion.')

    args = parser.parse_args()
    
    # init the openai model
    model = OpenAIModel()
    model_name = args.engine
    api_usage_tracker = APIUsageTracker()

    # load formatted trajectory
    work_path = pathlib.Path(args.data_path) / "trajectory" / args.agent_model_name / args.agent_type / f"ess{args.essential_count}_sen{args.sensitive_count}" 
    
    trajectory_path = work_path / "formatted_trajectory.json"
    with open(trajectory_path, 'r') as f:
        formatted_trajectories = json.load(f)
    
    def process_trajectory(trajectory):
        action = trajectory['final_action']
        # sensitive_entries = trajectory['sensitive_entries']
        sensitive_entries = zip(trajectory['sensitive_entries'], trajectory['sensitive_data'])
        essential_entries = zip(trajectory['essential_entries'], trajectory['essential_data'])
        outline = trajectory['case']['detailed_plot']
        leakage_rate_retrieved, leakage_rate_total, sensitive_retrieval_rate, sensitive_usage, sensitive_reasoning = measure_usage_rate("sensitive", outline, action, sensitive_entries, model, model_name)
        utility_rate_retrieved, utility_rate_total, essential_retrieval_rate, essential_usage, essential_reasoning = measure_usage_rate("essential", outline, action, essential_entries, model, model_name)
        trajectory['sensitive_usage'] = sensitive_usage
        trajectory['sensitive_reasoning'] = sensitive_reasoning
        trajectory['essential_usage'] = essential_usage
        trajectory['essential_reasoning'] = essential_reasoning
        trajectory['leakage_rate_retrieved'] = leakage_rate_retrieved
        trajectory['leakage_rate_total'] = leakage_rate_total
        trajectory['sensitive_retrieval_rate'] = sensitive_retrieval_rate
        trajectory['utility_rate_retrieved'] = utility_rate_retrieved
        trajectory['utility_rate_total'] = utility_rate_total
        trajectory['essential_retrieval_rate'] = essential_retrieval_rate

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_trajectory, trajectory) for trajectory in formatted_trajectories]
        for _ in tqdm(as_completed(futures), total=len(formatted_trajectories), desc="Processing trajectories"):
            pass
    
    output_filepath = work_path / f"evaluated_trajectory.json"
    with open(output_filepath, 'w') as f:
        json.dump(formatted_trajectories, f, indent=2)
    
    # leakage_rate_retrieved = sum(t['leakage_rate_retrieved'] for t in formatted_trajectories) / len(formatted_trajectories)
    # leakage_rate_total = sum(t['leakage_rate_total'] for t in formatted_trajectories) / len(formatted_trajectories)
    # sensitive_retrieval_rate = sum(t['sensitive_retrieval_rate'] for t in formatted_trajectories) / len(formatted_trajectories)
    # utility_rate_retrieved = sum(t['utility_rate_retrieved'] for t in formatted_trajectories) / len(formatted_trajectories)
    # utility_rate_total = sum(t['utility_rate_total'] for t in formatted_trajectories) / len(formatted_trajectories)
    # essential_retrieval_rate = sum(t['essential_retrieval_rate'] for t in formatted_trajectories) / len(formatted_trajectories)
    
    ...
    
    
if __name__ == "__main__":
    main()