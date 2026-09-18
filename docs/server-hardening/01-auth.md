# Opt-in bearer authentication

Start with `ft serve --model MODEL --api-key first,second` and send
`Authorization: Bearer first` (or `second`). Empty keys preserve existing desktop
behavior. Remote deployments should explicitly configure keys. Health probes are
public; admin endpoints still require the socket peer to be loopback. Other routes,
including metrics, docs and native generation, require authentication when enabled.
Keys are omitted from the configuration repr. CORS may handle a preflight before
authentication, while every actual request still crosses the auth boundary.

Like [vLLM's ASGI middleware](https://github.com/vllm-project/vllm/blob/v0.14.0/vllm/entrypoints/openai/api_server.py),
FreeToken compares fixed-size key digests without stopping at the first match and
does not buffer generation streams. Its exemption list is narrower: non-v1 routes
are protected too, while the existing loopback admin contract remains independent.
SGLang's route-level auth policies and TensorRT-LLM's broader serving/configuration
surface do not warrant a new configuration layer for this single-model frontend.
No generation path or protocol adapter is replaced.

Validation: `PYTHONPATH=python pytest -q tests/server/test_auth.py`.
These are CPU ASGI tests; they do not establish GPU or real-model behavior.
