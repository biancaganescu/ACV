agent_model_names=(
    "Qwen/Qwen2.5-32B-Instruct",
    "azure-DeepSeek-R1"
    "azure-DeepSeek-V3-0324"
    "azure-gpt-5-20250807"
    "azure-o3-20250416"
    "azure-gpt-4.1-20250414"
    "azure-grok-3"
    "azure-Kimi-K2-Thinking"
    "azure-gpt-4o-20241120"
)
simulator_model_name="azure-gpt-5.2-20251211"
i=4

for agent_model_name in "${agent_model_names[@]}"; do
    echo "Running experiment for agent: $agent_model_name"

    agent_temperature=0
    if [[ "$agent_model_name" == "azure-gpt-5-20250807" || "$agent_model_name" == "azure-o3-20250416" ]]; then
        agent_temperature=1
    fi

    python generate_trajectory.py \
        --input-path ../data/seed \
        --agent-model-name "$agent_model_name" \
        --agent-temperature $agent_temperature \
        --essential_count $i \
        --sensitive_count $i \
        --simulator-model-name "$simulator_model_name" \
        --critiquer-model-name "$simulator_model_name" \
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
