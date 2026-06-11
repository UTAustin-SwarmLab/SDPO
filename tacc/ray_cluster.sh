#!/bin/bash
# Start a multi-node Ray cluster inside a Slurm allocation.
# Not needed for single-node jobs: verl calls ray.init() locally in main_ppo.py.

start_ray_cluster_if_needed() {
    if [[ "${SLURM_NNODES:-1}" -le 1 ]]; then
        echo "Single-node job: skipping manual Ray startup (main_ppo will ray.init locally)."
        return 0
    fi

    local nodes
    nodes=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    readarray -t nodes_array <<< "$nodes"

    local head_node="${nodes_array[0]}"
    local head_node_ip
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    if [[ "$head_node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<< "$head_node_ip"
        if [[ ${#ADDR[0]} -gt 16 ]]; then
            head_node_ip=${ADDR[1]}
        else
            head_node_ip=${ADDR[0]}
        fi
        echo "IPv6 address detected; using IPv4 head address: $head_node_ip"
    fi

    local port=6379
    local ip_head="${head_node_ip}:${port}"
    export ip_head
    echo "Ray head address: $ip_head"

    echo "Starting Ray head on $head_node"
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        ray start --head --node-ip-address="$head_node_ip" --port="$port" \
        --num-cpus "${SLURM_CPUS_ON_NODE:-72}" --num-gpus "${GPUS_PER_NODE}" --block &
    sleep 10

    local worker_num=$((SLURM_JOB_NUM_NODES - 1))
    for ((i = 1; i <= worker_num; i++)); do
        local node_i="${nodes_array[$i]}"
        echo "Starting Ray worker $i on $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus "${SLURM_CPUS_ON_NODE:-72}" --num-gpus "${GPUS_PER_NODE}" --block &
        sleep 5
    done

    export RAY_HEAD_NODE="$head_node"
    echo "Ray cluster startup requested on ${SLURM_JOB_NUM_NODES} node(s)."
}

run_training_on_ray_head() {
    local run_cmd="$1"

    if [[ "${SLURM_NNODES:-1}" -le 1 ]]; then
        bash -lc "$run_cmd"
        return $?
    fi

    if [[ -z "${RAY_HEAD_NODE:-}" ]]; then
        echo "RAY_HEAD_NODE is unset; cannot launch training on Ray head." >&2
        return 1
    fi

    srun --overlap --nodes=1 --ntasks=1 -w "$RAY_HEAD_NODE" bash -lc "$run_cmd"
}
