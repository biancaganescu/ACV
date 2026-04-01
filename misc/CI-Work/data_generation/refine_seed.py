import json
import os
import sys
from argparse import ArgumentParser
import copy

import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import setup_prompt_environment, load_json_answer
from models.OpenAI import OpenAIModel, APIUsageTracker, Validator, bind_validator
import re
import pathlib
import random

prompt_env = setup_prompt_environment("./prompt_templates")
critique_template = prompt_env.get_template("critique_seed.j2")
refine_template = prompt_env.get_template("refine_seed.j2")

def prepare_critique_prompt(seed, candidates):
    system_prompt = critique_template.render(prompt_type="system")
    user_prompt = critique_template.render(prompt_type="user", seed=seed, candidates=candidates)
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]

def prepare_refine_prompt(seed, failed_essential, failed_sensitive):
    system_prompt = refine_template.render(prompt_type="system")
    user_prompt = refine_template.render(
        prompt_type="user", 
        seed=seed, 
        failed_essential=failed_essential,
        failed_sensitive=failed_sensitive
    )
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]

def refine_seed(seed, model, model_name, max_iterations=3):
    current_seed = copy.deepcopy(seed)
    
    for i in range(max_iterations):
        # 1. Prepare candidates for blind critique
        candidates_info = []
        for idx, item in enumerate(current_seed['essential_context']):
            candidates_info.append({'text': item, 'type': 'Essential', 'orig_idx': idx})
        for idx, item in enumerate(current_seed['sensitive_context']):
            candidates_info.append({'text': item, 'type': 'Sensitive', 'orig_idx': idx})
            
        random.shuffle(candidates_info)
        candidate_texts = [c['text'] for c in candidates_info]
        
        # 2. Critique
        prompt = prepare_critique_prompt(current_seed, candidate_texts)
        
        try:
            response = model.get_response_with_retry(
                input=prompt, 
                model=model_name,
                validator=lambda x: isinstance(load_json_answer(x), dict)
            )
            critique_text = model.extract_response_text(response)
            critique = load_json_answer(critique_text)
        except Exception as e:
            print(f"Error during critique step check: {e}")
            break

        reviews = critique.get('reviews', [])
        
        failed_essential = []
        failed_sensitive = []
        
        # Check reviews against original types
        for review in reviews:
            idx = review.get('index')
            reason = review.get('reason', 'No reason provided')
            predicted_label = review.get('label')
            
            if idx is not None and 0 <= idx < len(candidates_info):
                original_info = candidates_info[idx]
                original_type = original_info['type']
                original_idx = original_info['orig_idx']
                text = original_info['text']
                
                # Loose matching for labels to handle potential variation
                is_predicted_essential = 'essential' in str(predicted_label).lower()
                is_predicted_sensitive = 'sensitive' in str(predicted_label).lower()
                
                if original_type == 'Essential':
                    if not is_predicted_essential:
                        failed_essential.append({
                            'index': original_idx,
                            'text': text,
                            'reason': f"Evaluator classified as '{predicted_label}'. Reason: {reason}"
                        })
                elif original_type == 'Sensitive':
                    if not is_predicted_sensitive:
                        failed_sensitive.append({
                            'index': original_idx,
                            'text': text,
                            'reason': f"Evaluator classified as '{predicted_label}'. Reason: {reason}"
                        })
        
        if not failed_essential and not failed_sensitive:
            break # All passed
        
        # 3. Refine
        refine_prompt = prepare_refine_prompt(current_seed, failed_essential, failed_sensitive)
        try:
            response = model.get_response_with_retry(
                input=refine_prompt,
                model=model_name,
                validator=lambda x: isinstance(load_json_answer(x), dict)
            )
            refinement_text = model.extract_response_text(response)
            refinement = load_json_answer(refinement_text)
        except Exception as e:
            print(f"Error during refine step: {e}")
            break

        # Apply updates
        new_essentials = refinement.get('new_essential_contexts', [])
        new_sensitives = refinement.get('new_sensitive_contexts', [])
        
        updated = False
        for item in new_essentials:
            idx = item.get('index')
            if idx is not None and 0 <= idx < len(current_seed['essential_context']):
                current_seed['essential_context'][idx] = item['new_text']
                updated = True
                
        for item in new_sensitives:
            idx = item.get('index')
            if idx is not None and 0 <= idx < len(current_seed['sensitive_context']):
                current_seed['sensitive_context'][idx] = item['new_text']
                updated = True
        
        if not updated:
             break 

    return current_seed

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/seed", help='Path to the input seed jsonl file directory')
    parser.add_argument('--input_filename', type=str, default="enriched_seed.jsonl", help='Input filename')
    parser.add_argument('--output_filename', type=str, default="refined_seed.jsonl", help='Output filename')
    parser.add_argument('--essential_count', type=int, default=4, help='Configuration used for path')
    parser.add_argument('--sensitive_count', type=int, default=4, help='Configuration used for path')
    parser.add_argument('--engine', type=str, default='gpt-5.2-20251211', help='The language model engine.')
    parser.add_argument('--overwrite', default=True, help='Whether to overwrite existing output file.')
    args = parser.parse_args()
    
    # Construct paths
    base_dir = pathlib.Path(args.data_path) / f"ess{args.essential_count}_sen{args.sensitive_count}"
    seed_filepath = base_dir / args.input_filename
    output_filepath = base_dir / args.output_filename
    
    if not seed_filepath.exists():
        print(f"Input file not found: {seed_filepath}")
        if (pathlib.Path(args.data_path) / args.input_filename).exists():
             seed_filepath = pathlib.Path(args.data_path) / args.input_filename
             output_filepath = pathlib.Path(args.data_path) / args.output_filename
        else:
             return

    print(f"Reading from {seed_filepath}")
    
    with open(seed_filepath, 'r') as f:
        seeds = [json.loads(line) for line in f]
    
    model = OpenAIModel()
    MODEL_IDENTIFIER = args.engine
    api_usage_tracker = APIUsageTracker()
    
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_filepath.exists():
        output_filepath.unlink()
        print(f"Overwriting {output_filepath}")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(refine_seed, seed, model, MODEL_IDENTIFIER) for seed in seeds]
        for future in tqdm(as_completed(futures), total=len(seeds), desc="Refining Seeds"):
            try:
                refined_seed = future.result()
                with open(output_filepath, 'a') as f:
                    f.write(json.dumps(refined_seed) + '\n')
            except Exception as e:
                print(f"Error processing seed: {e}")

    print(f"Refinement complete. Saved to {output_filepath}")

if __name__ == '__main__':
    main()
