agent_model_name="azure-gpt-5-20250807"
simulator_model_name="azure-gpt-5.2-20251211"
i=4
agent_temperature=1


for agent_type in ci_cot privacy_enhanced ; do

    python generate_trajectory.py \
        --input-path ../data/seed \
        --agent-model-name "$agent_model_name" \
        --agent-temperature $agent_temperature \
        --essential_count $i \
        --sensitive_count $i \
        --simulator-model-name "$simulator_model_name" \
        --critiquer-model-name "$simulator_model_name" \
        --dump-dir ../data/trajectory \
        --agent-type $agent_type
done