agent_model_name="azure-gpt-5-20250807"
simulator_model_name="azure-gpt-5.2-20251211"
i=4
agent_temperature=1

for pressure_type in regular intentional unintentional; do
    python outline2instruction.py --essential_count $i --sensitive_count $i --pressure_type $pressure_type

    python generate_trajectory.py \
        --pressure_type $pressure_type \
        --input-path ../data/seed \
        --agent-model-name "$agent_model_name" \
        --agent-temperature $agent_temperature \
        --essential_count $i \
        --sensitive_count $i \
        --simulator-model-name "$simulator_model_name" \
        --critiquer-model-name "$simulator_model_name" \
        --dump-dir ../data/trajectory
done