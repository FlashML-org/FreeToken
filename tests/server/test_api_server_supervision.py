"""Production-path coverage for backend-supervisor ownership wiring."""

from __future__ import annotations

from types import SimpleNamespace  # Build minimal production configuration and process fixtures.

import freetoken.server.api_server as api_server  # Exercise run_api_server rather than a duplicate helper.
import freetoken.server.supervisor as supervisor  # Compare the captured thread target with production supervision.


def test_run_api_server_records_handle_before_starting_supervisor(monkeypatch):  # Prove production ownership reaches shutdown state.
    created_threads = []  # Capture the daemon thread without running its infinite liveness loop.

    class FakeThread:  # Model the threading.Thread constructor and start surface used by run_api_server.
        def __init__(self, *, target, args, kwargs, name, daemon) -> None:  # Preserve every wiring argument for assertions.
            self.target = target  # Record the production supervisor callable.
            self.args = args  # Record the backend handle, progress object, and ready callback.
            self.kwargs = kwargs  # Record failure, metadata, and shutdown callbacks.
            self.name = name  # Preserve the operational thread identity.
            self.daemon = daemon  # Preserve process-exit behavior.
            self.started = False  # Begin before the production start call.
            created_threads.append(self)  # Publish the constructed thread to the test.

        def start(self) -> None:  # Mirror the thread start surface without executing the target.
            self.started = True  # Make production ordering observable.

    handle = SimpleNamespace(processes=[SimpleNamespace(name="worker")])  # Supply the launch ownership returned by start_backend.
    config = SimpleNamespace(  # Supply only fields read by the production wiring path.
        sampling_defaults="framework",  # Avoid checkpoint sampling reads.
        use_dummy_weight=True,  # Keep the test independent of model artifacts.
        server_host="127.0.0.1",  # Provide a deterministic local endpoint.
        server_port=1919,  # Provide a deterministic test port without binding it.
        cors_origins=[],  # Preserve the CORS installation contract.
        served_model_name="test-model",  # Satisfy FrontendManager configuration consumers.
        zmq_frontend_addr="inproc://frontend",  # Supply the receive queue address.
        zmq_tokenizer_addr="inproc://tokenizer",  # Supply the send queue address.
        frontend_create_tokenizer_link=True,  # Preserve the queue creation flag.
    )  # Complete the minimal server configuration.
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", None)  # Start from a fresh production singleton.
    api_server._SHUTTING_DOWN.clear()  # Remove shutdown state left by another test.
    monkeypatch.setattr(api_server, "ZmqAsyncPullQueue", lambda *args, **kwargs: SimpleNamespace(stop=lambda: None))  # Avoid real ZMQ endpoints.
    monkeypatch.setattr(api_server, "ZmqAsyncPushQueue", lambda *args, **kwargs: SimpleNamespace(stop=lambda: None))  # Avoid real ZMQ endpoints.
    monkeypatch.setattr(api_server, "install_cors", lambda *_args, **_kwargs: None)  # Avoid mutating the shared FastAPI app.
    monkeypatch.setattr(api_server, "init_request_logging", lambda: None)  # Avoid starting a writer thread.
    monkeypatch.setattr(api_server, "install_polling_access_log_filter", lambda: None)  # Avoid global logging changes.
    monkeypatch.setattr(api_server.threading, "Thread", FakeThread)  # Capture production supervisor construction.
    monkeypatch.setattr(api_server.uvicorn, "run", lambda *_args, **_kwargs: None)  # Stop after wiring without binding a socket.

    api_server.run_api_server(config, lambda: handle, run_shell=False)  # Execute the complete production wiring path.

    thread = created_threads[0]  # Inspect the sole backend-supervisor thread.
    assert api_server._GLOBAL_STATE.backend_handle is handle  # Require the queue-owning handle to survive for shutdown.
    assert api_server._GLOBAL_STATE.backend_processes == handle.processes  # Require child ownership to survive for teardown.
    assert api_server._GLOBAL_STATE.backend_supervisor_thread is thread  # Require shutdown to retain the actual thread object.
    assert thread.target is supervisor.run_backend_supervisor  # Require the production supervisor implementation.
    assert thread.args[0] is handle  # Require the returned launch handle to reach the supervisor unchanged.
    assert thread.started is True  # Require ownership publication before production starts supervision.
