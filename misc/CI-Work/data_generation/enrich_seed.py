
import json
import re
import sys
from utils import setup_prompt_environment, load_json_answer
from argparse import ArgumentParser
from models.OpenAI import OpenAIModel, APIUsageTracker, Validator, bind_validator
from tqdm import tqdm
import pathlib
import uuid


prompt_env= setup_prompt_environment("./prompt_templates")
prompt_template = prompt_env.get_template("enrich_seed.j2")
    
def prepare_enrich_seed_prompt(
        raw_seed,
        essential_count,
        sensitive_count
):  
    system_prompt = prompt_template.render(prompt_type="system")
    user_prompt = prompt_template.render(
        prompt_type="user",
        raw_seed=json.dumps(raw_seed),
        num_essential=essential_count,
        num_sensitive=sensitive_count,
    )
    prompt=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    return prompt 
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, default="../data/seed/raw_seed.jsonl", help='Path to the input seed jsonl file')
    parser.add_argument('--output_path', type=str, default="../data/seed", help='Path to the output enriched seed jsonl file')
    parser.add_argument('--essential_count', type=int, default=10, help='Number of essential data points to enrich')
    parser.add_argument('--sensitive_count', type=int, default=10, help='Number of sensitive data points to enrich')
    parser.add_argument('--engine', type=str, default='gpt-5.2-20251211', help='The language model engine to use for completion.')
    parser.add_argument('--overwrite', default=True, help='Whether to overwrite existing output file.')
    args = parser.parse_args()
    
    # load input seeds
    with open(args.input_file, 'r') as f:
        seeds = [json.loads(line) for line in f]
    # sort seeds by type
    seeds.sort(key=lambda x: x['type'])
    # prepare prompt list
    prompt_list = [prepare_enrich_seed_prompt(seed, args.essential_count, args.sensitive_count) for seed in seeds]
    
    # init the openai model
    model = OpenAIModel()
    MODEL_IDENTIFIER = args.engine
    api_usage_tracker = APIUsageTracker()
    
    @bind_validator
    def extract_enrichseed_answer(answer):
        raw_result = load_json_answer(answer)
        assert len(raw_result['essential_context']) == args.essential_count
        assert len(raw_result['sensitive_context']) == args.sensitive_count
        assert len(raw_result['sensitive_category']) == args.sensitive_count
        return raw_result
    
    responses = model.get_multi_response(
         inputs=prompt_list,
         model=MODEL_IDENTIFIER,
         max_workers=36,
         validator=extract_enrichseed_answer.validate,
    )
    
    bar = tqdm(enumerate(responses), total=len(prompt_list), desc="Enriching Seeds")
    
    output_filepath = pathlib.Path(args.output_path) / f"ess{args.essential_count}_sen{args.sensitive_count}" / "enriched_seed.jsonl"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_filepath.exists():
        output_filepath.unlink()
        print(f"Overwriting existing file at {output_filepath}")
    for idx, response in bar:
        enrich_seed = extract_enrichseed_answer(model.extract_response_text(response))
        enrich_seed['uuid'] = str(uuid.uuid4())
        enrich_seed['type'] = seeds[idx]['type']
        with open(output_filepath, 'a') as f:
            f.write(json.dumps(enrich_seed) + '\n')
        
        # ambiguous_entries = pd.DataFrame(amb_entry)
        # remove_invalid_entries(owner, outsider, ambiguous_entries)
        # ambiguous_entries = ambiguous_entries.assign(owner=owner, outsider=outsider)
        # ambiguous_entries.to_csv(
        #     output_filepath,
        #     mode="a",
        #     header=write_header,
        #     index=False,
        # )
        # write_header = False
        # total_tokens = sum(api_usage_tracker.get_model_usage(MODEL_IDENTIFIER).values())
        # bar.set_postfix({
        #     "#Tokens": api_usage_tracker.format_token_count(total_tokens), 
        #     "#Entries": len(ambiguous_entries),
        # })
        
if __name__ == "__main__":
    main()