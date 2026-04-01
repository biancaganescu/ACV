"""Convert the privacy-sensitive seeds into a vignette via template-based generation.
Expected input format:
[
    {
        "name": "",
        "seed: {
            "data_type": "",
            "data_subject": "",
            "data_sender": "",
            "data_sender_name": "",
            "data_recipient": "",
            "transmission_principle": "",
            "source": ""
        }
    },
    ...
]
Expected output format:
[
    {
        "name": "",
        "seed": {...}
        "vignette": {
            "story": "",
            "data_type_concrete": "",
            "data_subject_concrete": "",
            "data_sender_concrete": "",
            "data_recipient_concrete": "",
        }
    }
]
"""
import json
import os
import sys
from argparse import ArgumentParser

import openai
from tqdm import tqdm

from utils import openai_chat_completion_with_retry, SurgeryKitUnitTest, SurgeryKitModule, setup_prompt_environment
from models.OpenAI import OpenAIModel, APIUsageTracker, Validator, bind_validator
import re
import pathlib

prompt_env= setup_prompt_environment("./prompt_templates")
prompt_template = prompt_env.get_template("seed2outline.j2")

@bind_validator
def parse_seed2outline_answer(answer: str):
    section_pat = re.compile(r'\[(Outline|Sensitive Data|Essential Data)\]:', re.IGNORECASE)
    parts = section_pat.split(answer)
    parsed = {
        'outline': '',
        'sensitive_data': [],
        'essential_data': []
    }

    def parse_data_section(text: str):
        items = []
        current = None
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r'\d+\.\s+\*\*Input Abstract:\*\*\s*(.+)', line)
            if m:
                if current:
                    items.append(current)
                current = {
                    'Input Abstract': m.group(1).strip(),
                    # 'Specific Detail': '',
                    'Source': ''
                }
                continue
            # m = re.match(r'\*\s+\*\*Specific Detail:\*\*\s*(.+)', line)
            # if m and current is not None:
            #     current['Specific Detail'] = m.group(1).strip()
            #     continue
            m = re.match(r'\*\s+\*\*Source:\*\*\s*(.+)', line)
            if m and current is not None:
                current['Source'] = m.group(1).strip()
                continue
        if current:
            items.append(current)
        return items

    for i in range(1, len(parts), 2):
        name = parts[i].lower()
        content = parts[i + 1].strip()
        if name == 'outline':
            parsed['outline'] = content
        elif name == 'sensitive data':
            parsed['sensitive_data'] = parse_data_section(content)
        elif name == 'essential data':
            parsed['essential_data'] = parse_data_section(content)

    return parsed

def test_no_explicit_word(instruction, output):
    story = output.split('\n')[0]
    if 'sensitive' in story.lower() or 'private' in story.lower() or 'privacy' in story.lower() or 'confident' in story.lower() or 'secret' in story.lower():
        return False, ['Remove words that explicitly state sensitivity without changing anything else.']
    return True, []


test_no_explicit_word_unit_test = SurgeryKitUnitTest(
    name='test_no_explicit_word',
    description='The generated vignette should not contain explicit words like "confident", "sensitive", "private" that'
                ' indicate sensitivity.',
    test_func=test_no_explicit_word
)


def prepare_args():
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path of the privacy-sensitive seeds in json format.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the generated vignettes in json format.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start index of the seeds to process. If -1, process all remaining seeds.')
    parser.add_argument('--num', type=int, default=1,
                        help='Number of seeds to process.')
    parser.add_argument('--specific-case-name', type=str, default=None,
                        help='If not None, only process the case with the given name.')
    parser.add_argument('--use-surgery-kit', action='store_true',
                        help='Whether to use the surgery kit to refine the output.')
    parser.add_argument('--surgery-kit-max-try', type=int, default=2,
                        help='The maximum number of attempts to refine the output.')
    parser.add_argument('--engine', type=str, default='gpt-4-1106',
                        help='The language model engine to use for completion.')

    return parser.parse_args()


def run_seed_to_outline(
        case_idx,
        seed,
        # data_subject,
        # data_sender,
        # data_sender_name,
        # data_recipient,
        # data_sender_task,
        # essential_context,
        # sensitive_context,
        engine,
        surgery_kit=None
):
    
    prompt = prepare_seed2outline_prompt(seed)
    response = openai_chat_completion_with_retry(
        engine=engine,
        messages=prompt,
        # max_tokens=1000,
        temperature=0.0
    )
    answer = response.choices[0].message['content'].strip()

    try:
        sample = parse_seed2outline_answer(answer)
        # if surgery_kit:
        #     refined_output, refine_round = surgery_kit.run(
        #         instruction=prompt,
        #         original_output=answer,
        #         unit_tests=[test_no_explicit_word_unit_test]
        #     )
        #     sample['vignette']['refine_round'] = refine_round
        #     sample['vignette']['story'] = refined_output.split('\n')[0].replace('[Vignette]: ', '').strip()
        #     sample['vignette']['story_before_refinement'] = lines[0].replace('[Vignette]: ', '').strip()

        return sample

    except Exception as e:
        print(f'Error {e} occurred when processing {case_idx}')
        return None

def prepare_seed2outline_prompt(
    seed
):
    system_prompt = prompt_template.render(prompt_type="system")
    user_prompt = prompt_template.render(
        prompt_type="user",
        seed=seed,
        tools = ["Zoom", "Calendar", "Slack", "Notion", "Email"]
    )
    prompt = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    return prompt

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/seed", help='Path to the input seed jsonl file')
    parser.add_argument('--essential_count', type=int, default=2, help='Number of essential data points to enrich')
    parser.add_argument('--sensitive_count', type=int, default=2, help='Number of sensitive data points to enrich')
    parser.add_argument('--engine', type=str, default='gpt-5.2-20251211', help='The language model engine to use for completion.')
    parser.add_argument('--overwrite', default=True, help='Whether to overwrite existing output file.')
    args = parser.parse_args()
    
    seed_filepath = pathlib.Path(args.data_path) / f"ess{args.essential_count}_sen{args.sensitive_count}" / "refined_seed.jsonl"
    
    # load input seeds
    with open(seed_filepath, 'r') as f:
        seeds = [json.loads(line) for line in f]
    # prepare prompt list
    prompt_list = [prepare_seed2outline_prompt(seed) for seed in seeds]

    # init the openai model
    model = OpenAIModel()
    MODEL_IDENTIFIER = args.engine
    api_usage_tracker = APIUsageTracker()
    
    responses = model.get_multi_response(
         inputs=prompt_list,
         model=MODEL_IDENTIFIER,
         max_workers=36,
         validator=parse_seed2outline_answer.validate,
    )
    
    outline_filepath = pathlib.Path(args.data_path) / f"ess{args.essential_count}_sen{args.sensitive_count}" / "outline_seed.jsonl"
    outline_filepath.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and outline_filepath.exists():
        outline_filepath.unlink()
        print(f"Overwriting existing file at {outline_filepath}")
    bar = tqdm(enumerate(responses), total=len(prompt_list), desc="Seed2Outline")
    
    for idx, response in bar:
        outline = parse_seed2outline_answer(model.extract_response_text(response))
        outline['uuid'] = seeds[idx]['uuid']
        outline['type'] = seeds[idx]['type']
        outline['sensitive_category'] = seeds[idx]['sensitive_category']
        with open(outline_filepath, 'a') as f:
            f.write(json.dumps(outline) + '\n')


if __name__ == '__main__':
    main()
