import os
os.environ.update({
 'RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR':'',
 'RAY_SERVE_ENABLE_HA_PROXY':'1',
 'RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING':'1',
 'RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY':'1',
 'RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE':'8388608',
})
import ray
ray.init(address='auto',ignore_reinit_error=True,runtime_env={'env_vars':{
 'PYTHONPATH':'/tmp/online-prefill-overlay:/home/ray/default/online-prefill-harness/src',
 'ONLINE_CALIBRATION_OVERLAY_REV':'3',
 'RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR':'',
 **{k:v for k,v in os.environ.items() if k in ('RAY_SERVE_ENABLE_HA_PROXY','RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING','RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY','RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE')}
}})
from serve_llm_autoscaling.cli import main
raise SystemExit(main(['run','experiments/h100_16sessions_online_queue_1s.yaml']))
