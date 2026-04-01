


agent_model_name="azure-gpt-5-20250807"
agent_temperature=1
for i in 1 2 4 6 8 10 16
do
    python enrich_seed.py --essential_count $i --sensitive_count $i
    python refine_seed.py --essential_count $i --sensitive_count $i
    python seed2outline.py --essential_count $i --sensitive_count $i
    python outline2instruction.py --essential_count $i --sensitive_count $i

    python generate_trajectory.py \
        --input-path ../data/seed \
        --agent-model-name "$agent_model_name" \
        --agent-temperature $agent_temperature \
        --essential_count $i \
        --sensitive_count $i \
        --simulator-model-name azure-gpt-5.2-20251211 \
        --critiquer-model-name azure-gpt-5.2-20251211 \
        --dump-dir ../data/trajectory

    python format_trajectory.py \
        --data_path ../data \
        --agent_model_name "$agent_model_name" \
        --agent_type naive \
        --essential_count $i \
        --sensitive_count $i 

    python evaluate_trajectory.py \
        --data_path ../data \
        --agent_model_name "$agent_model_name" \
        --agent_type naive \
        --essential_count $i \
        --sensitive_count $i

done