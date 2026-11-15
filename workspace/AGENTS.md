# Workspace context

Before starting Ray Serve/vLLM experiments or restoring this workspace, read
[required Ray/vLLM runtime patches](agent-context/ray-vllm-runtime-patches.md).
The H100 setup needs the preserved device-aware compilation-cache patch for
vLLM 0.33; otherwise additional replicas can fail startup while the benchmark
continues on one replica. Verify the patch after package reinstalls or restarts,
and validate two healthy model replicas before an autoscaling experiment.

For online prefill queue learning, uncached token workload regression, queue-time
definitions, or router-versus-trace RMSE comparisons, read
[online prefill queue learning context](agent-context/online-prefill-queue-learning.md).
It records the recovered checkouts, results, limitations, artifacts, and commands.

Repository-specific instructions still apply within each repository.
