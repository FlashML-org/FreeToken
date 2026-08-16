from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_SERVER = "http://127.0.0.1:1919"
CODEX_PROFILE = "freetoken-launch"
CODEX_PROVIDER_NAME = "FreeToken"
CODEX_CATALOG_NAME = "freetoken-model.json"
CODEX_PROVIDER_API_KEY_ENV = "FREETOKEN_API_KEY"
# Used when the server reports no context length. Guessing low only costs earlier compaction.
FALLBACK_CONTEXT_WINDOW = 128_000
MAX_OUTPUT_TOKENS_CAP = 32_768
OPENCODE_PROVIDER = "freetoken"
OPENCODE_PROVIDER_NAME = "FreeToken"
OPENCLAW_PROVIDER = "freetoken"
OPENCLAW_PROVIDER_NAME = "FreeToken"
HERMES_API_KEY = "freetoken-local"
HERMES_MIN_CONTEXT_LENGTH = 64_000  # Hermes refuses to start below this
DSH_API_KEY = "freetoken"
DSH_MIN_NODE = (22, 19)  # dsh's floor; its npm package declares no engines gate
DSH_LAUNCH_SETTINGS_NAME = "freetoken-launch.settings.yaml"
DSH_LAUNCH_PATCH_NAME = "freetoken-launch.cordis.patch.yml"
CLOUD_PROVIDER_API_KEY_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
)
CODEX_CLEAR_ENV = (*CLOUD_PROVIDER_API_KEY_ENV, "CODEX_API_KEY")
OPENCLAW_CLEAR_ENV = CLOUD_PROVIDER_API_KEY_ENV


@dataclass(frozen=True)
class ServerURL:
    origin: str
    openai_base_url: str


@dataclass(frozen=True)
class ServedModel:
    model_id: str
    models: list[str]
    # The model's own limit, not the KV budget in force: a rebuild moves that, and agents read
    # their config only at startup. None when the server does not report one.
    context_length: int | None = None


@dataclass(frozen=True)
class LaunchContext:
    server: ServerURL
    model: ServedModel
    extra_args: list[str]
    dry_run: bool
    assume_yes: bool = False


@dataclass(frozen=True)
class CommandSpec:
    argv: list[str]
    env: dict[str, str]
    unset_env: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentInstaller:
    display_name: str
    argv: tuple[str, ...]
    dependencies: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()


AGENT_INSTALLERS = {
    "codex": AgentInstaller(
        display_name="Codex CLI",
        argv=("sh", "-c", "curl -fsSL https://chatgpt.com/codex/install.sh | sh"),
        dependencies=("curl", "sh"),
        env=(("CODEX_NON_INTERACTIVE", "1"),),
    ),
    "claude": AgentInstaller(
        display_name="Claude Code",
        argv=("bash", "-c", "curl -fsSL https://claude.ai/install.sh | bash"),
        dependencies=("curl", "bash"),
    ),
    "opencode": AgentInstaller(
        display_name="OpenCode",
        argv=("bash", "-c", "set -o pipefail; curl -fsSL https://opencode.ai/install | bash"),
        dependencies=("curl", "bash"),
    ),
    "openclaw": AgentInstaller(
        display_name="OpenClaw",
        argv=("bash", "-c", "curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard"),
        dependencies=("curl", "bash"),
    ),
    "hermes": AgentInstaller(
        display_name="Hermes",
        argv=(
            "bash",
            "-lc",
            "curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-setup --non-interactive",
        ),
        dependencies=("bash", "curl", "git"),
    ),
    "dsh": AgentInstaller(
        display_name="DeepSeek Harness",
        argv=("npm", "install", "--global", "@deepseek-ai/dsh"),
        dependencies=("npm", "node"),
    ),
}


def _format_netloc(host: str, port: int | None) -> str:
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is None:
        return host
    return f"{host}:{port}"


def _connectable_host(host: str) -> str:
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def resolve_server_url(server: str | None) -> ServerURL:
    raw = server or os.environ.get("FREETOKEN_HOST") or DEFAULT_SERVER
    raw = raw.strip()
    if not raw:
        raw = DEFAULT_SERVER
    if "://" not in raw:
        raw = f"http://{raw}"

    parsed = urlsplit(raw)
    if not parsed.hostname:
        raise ValueError(f"Invalid FreeToken server URL: {server!r}")

    path = parsed.path.rstrip("/")
    if path not in ("", "/v1"):
        raise ValueError(
            "FreeToken launch --server accepts only an origin or /v1 base URL"
        )

    host = _connectable_host(parsed.hostname)
    netloc = _format_netloc(host, parsed.port)
    origin = urlunsplit((parsed.scheme or "http", netloc, "", "", "")).rstrip("/")
    return ServerURL(origin=origin, openai_base_url=f"{origin}/v1")


def _get_json(url: str) -> object:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _stats_context_length(server: ServerURL) -> int | None:
    """Fallback for servers whose /v1/models predates the context fields."""
    try:
        payload = _get_json(f"{server.origin}/v1/stats")
    except (HTTPError, URLError, OSError, TimeoutError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    return _positive_int(model.get("ctx")) if isinstance(model, dict) else None


def discover_server_model(server: ServerURL) -> ServedModel:
    try:
        _get_json(server.openai_base_url)
        payload = _get_json(f"{server.openai_base_url}/models")
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        raise RuntimeError(f"Cannot connect to FreeToken server at {server.origin}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("FreeToken server returned an invalid /v1/models response")
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("FreeToken server returned an invalid /v1/models response")

    models: list[str] = []
    context_length: int | None = None
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        if not models:
            context_length = _positive_int(item.get("max_model_len")) or _positive_int(
                item.get("context_length")
            )
        models.append(item["id"])

    if not models:
        raise RuntimeError(f"FreeToken server at {server.origin} reported no models")

    if context_length is None:
        context_length = _stats_context_length(server)

    return ServedModel(model_id=models[0], models=models, context_length=context_length)


def _backup_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{time.time_ns()}.bak")


def _write_text_with_backup(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        shutil.copy2(path, _backup_path(path))
    path.write_text(text)


def _write_json_with_backup(path: Path, value: object) -> None:
    _write_text_with_backup(path, json.dumps(value, indent=2) + "\n")


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _remove_toml_section(text: str, header: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped == header:
            skipping = True
            continue
        if skipping and stripped.startswith("[") and stripped.endswith("]"):
            skipping = False
        if not skipping:
            kept.append(line)
    result = "\n".join(kept).strip()
    return f"{result}\n\n" if result else ""


def _codex_config_path() -> Path:
    return Path.home() / ".codex" / "config.toml"


def _codex_profile_path() -> Path:
    return Path.home() / ".codex" / f"{CODEX_PROFILE}.config.toml"


def _codex_catalog_path() -> Path:
    return _codex_profile_path().with_name(CODEX_CATALOG_NAME)


def _context_window(ctx: LaunchContext) -> int:
    return ctx.model.context_length or FALLBACK_CONTEXT_WINDOW


def _max_output_tokens(ctx: LaunchContext) -> int:
    """Scales with the window: agents derive their compaction headroom from this."""
    return max(1, min(_context_window(ctx) // 4, MAX_OUTPUT_TOKENS_CAP))


def _codex_catalog_entry(model_id: str, window: int) -> dict[str, object]:
    return {
        "slug": model_id,
        "display_name": model_id,
        "description": model_id,
        "context_window": window,
        "max_context_window": window,
        "effective_context_window_percent": 100,
        "shell_type": "default",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 0,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "input_modalities": ["text"],
        "base_instructions": "",
        "default_reasoning_summary": "none",
        "supported_reasoning_levels": [],
        "supports_reasoning_summaries": False,
        "experimental_supported_tools": [],
        "supports_search_tool": False,
        "web_search_tool_type": "text",
        "supports_image_detail_original": False,
        "support_verbosity": True,
        "default_verbosity": "low",
        "supports_parallel_tool_calls": False,
        "use_responses_lite": False,
        "model_messages": None,
        "upgrade": None,
    }


def _ordered_model_ids(ctx: LaunchContext) -> list[str]:
    result: list[str] = []
    for model_id in [ctx.model.model_id, *ctx.model.models]:
        if model_id and model_id not in result:
            result.append(model_id)
    return result


def _remove_toml_root_string_value(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    in_root = True
    expected = _toml_string(value)

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_root = False
        if in_root and "=" in stripped:
            lhs, rhs = stripped.split("=", 1)
            rhs = rhs.split("#", 1)[0].strip()
            if lhs.strip() == key and rhs == expected:
                continue
        kept.append(line)

    result = "\n".join(kept).strip()
    return f"{result}\n" if result else ""


def _codex_migrated_base_config_text(existing: str) -> str:
    text = _remove_toml_section(existing, f"[profiles.{CODEX_PROFILE}]")
    text = _remove_toml_section(text, f"[model_providers.{CODEX_PROFILE}]")
    return _remove_toml_root_string_value(text, "profile", CODEX_PROFILE)


def _codex_profile_text(ctx: LaunchContext, catalog_path: Path) -> str:
    return (
        f"model = {_toml_string(ctx.model.model_id)}\n"
        + f"openai_base_url = {_toml_string(ctx.server.openai_base_url)}\n"
        + f"model_provider = {_toml_string(CODEX_PROFILE)}\n"
        + f"model_catalog_json = {_toml_string(str(catalog_path))}\n\n"
        + f"[model_providers.{CODEX_PROFILE}]\n"
        + f"name = {_toml_string(CODEX_PROVIDER_NAME)}\n"
        + f"base_url = {_toml_string(ctx.server.openai_base_url)}\n"
        + 'wire_api = "responses"\n'
        + f"env_key = {_toml_string(CODEX_PROVIDER_API_KEY_ENV)}\n"
    )


def prepare_codex(ctx: LaunchContext) -> CommandSpec:
    catalog_path = _codex_catalog_path()
    if not ctx.dry_run:
        config_path = _codex_config_path()
        profile_path = _codex_profile_path()
        window = _context_window(ctx)
        catalog = {
            "models": [
                _codex_catalog_entry(model_id, window) for model_id in _ordered_model_ids(ctx)
            ],
        }
        _write_json_with_backup(catalog_path, catalog)
        _write_text_with_backup(profile_path, _codex_profile_text(ctx, catalog_path))

        if config_path.exists():
            existing = config_path.read_text()
            migrated = _codex_migrated_base_config_text(existing)
            if migrated != existing:
                _write_text_with_backup(config_path, migrated)

    argv = [
        "codex",
        "--profile",
        CODEX_PROFILE,
        "-c",
        f"model_catalog_json={_toml_string(str(catalog_path))}",
        "-m",
        ctx.model.model_id,
        *ctx.extra_args,
    ]
    return CommandSpec(
        argv=argv,
        env={CODEX_PROVIDER_API_KEY_ENV: "freetoken"},
        unset_env=CODEX_CLEAR_ENV,
    )


def prepare_claude(ctx: LaunchContext) -> CommandSpec:
    model_id = ctx.model.model_id
    return CommandSpec(
        argv=["claude", "--model", model_id, *ctx.extra_args],
        env={
            # Honoured only for model ids not starting with "claude-"; else it assumes 200k.
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(_context_window(ctx)),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(_max_output_tokens(ctx)),
            "ANTHROPIC_BASE_URL": ctx.server.origin,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "freetoken",
            "ANTHROPIC_MODEL": model_id,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model_id,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model_id,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_id,
            "CLAUDE_CODE_SUBAGENT_MODEL": model_id,
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        },
    )


def _opencode_state_path() -> Path:
    return Path.home() / ".local" / "state" / "opencode" / "model.json"


def _opencode_model_entries(
    model_ids: list[str], window: int, output: int
) -> dict[str, dict[str, object]]:
    # Without `limit.context` OpenCode reads the window as 0 and stops auto-compacting.
    return {
        model_id: {"name": model_id, "limit": {"context": window, "output": output}}
        for model_id in model_ids
    }


def _opencode_config(ctx: LaunchContext) -> str:
    model_ids = _ordered_model_ids(ctx)
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            OPENCODE_PROVIDER: {
                "npm": "@ai-sdk/openai-compatible",
                "name": OPENCODE_PROVIDER_NAME,
                "options": {"baseURL": ctx.server.openai_base_url},
                "models": _opencode_model_entries(
                    model_ids, _context_window(ctx), _max_output_tokens(ctx)
                ),
            },
        },
        "model": f"{OPENCODE_PROVIDER}/{ctx.model.model_id}",
    }
    return json.dumps(config)


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if isinstance(value, dict):
        return value
    return {}


def _update_opencode_state(path: Path, model_ids: list[str]) -> None:
    state = _load_json_object(path)
    recent = state.get("recent")
    if not isinstance(recent, list):
        recent = []

    model_set = set(model_ids)
    new_recent: list[object] = []
    for entry in recent:
        if (
            isinstance(entry, dict)
            and entry.get("providerID") == OPENCODE_PROVIDER
            and entry.get("modelID") in model_set
        ):
            continue
        new_recent.append(entry)

    for model_id in reversed(model_ids):
        new_recent.insert(0, {"providerID": OPENCODE_PROVIDER, "modelID": model_id})

    state.setdefault("favorite", [])
    state.setdefault("variant", {})
    state["recent"] = new_recent[:10]
    _write_json_with_backup(path, state)


def prepare_opencode(ctx: LaunchContext) -> CommandSpec:
    if not ctx.dry_run:
        _update_opencode_state(_opencode_state_path(), _ordered_model_ids(ctx))
    return CommandSpec(
        argv=["opencode", *ctx.extra_args],
        env={"OPENCODE_CONFIG_CONTENT": _opencode_config(ctx)},
    )


def _openclaw_config_path() -> Path:
    return Path.home() / ".openclaw" / "openclaw.json"


def _openclaw_sessions_path() -> Path:
    return Path.home() / ".openclaw" / "agents" / "main" / "sessions" / "sessions.json"


def _dict_child(parent: dict[str, object], key: str) -> dict[str, object]:
    child = parent.get(key)
    if isinstance(child, dict):
        return child
    child = {}
    parent[key] = child
    return child


def _openclaw_model_entry(model_id: str, window: int, output: int) -> dict[str, object]:
    return {
        "id": model_id,
        "name": model_id,
        "input": ["text"],
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
        },
        "contextWindow": window,
        "maxTokens": output,
    }


def _openclaw_has_freetoken_provider(config: dict[str, object]) -> bool:
    models_section = config.get("models")
    if not isinstance(models_section, dict):
        return False
    providers = models_section.get("providers")
    if not isinstance(providers, dict):
        return False
    return isinstance(providers.get(OPENCLAW_PROVIDER), dict)


def _confirm_openclaw_first_patch(ctx: LaunchContext, config_path: Path) -> bool:
    print(
        "FreeToken launch will modify OpenClaw config for the first time.",
        file=sys.stderr,
    )
    print(f"  File: {config_path}", file=sys.stderr)
    print(f"  Provider: {OPENCLAW_PROVIDER}", file=sys.stderr)
    print(f"  Model: {OPENCLAW_PROVIDER}/{ctx.model.model_id}", file=sys.stderr)
    print(f"  Base URL: {ctx.server.openai_base_url}", file=sys.stderr)
    if ctx.assume_yes:
        return True
    if not sys.stdin.isatty():
        print("OpenClaw config patch cancelled: stdin is not interactive.", file=sys.stderr)
        return False
    while True:
        answer = input("Continue and update OpenClaw config? [y/n] ").strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False
        print("Please answer y or n.", file=sys.stderr)


def _patch_openclaw_config(
    ctx: LaunchContext,
    config: dict[str, object],
) -> dict[str, object]:
    models_section = _dict_child(config, "models")
    providers = _dict_child(models_section, "providers")
    provider = providers.get(OPENCLAW_PROVIDER)
    if not isinstance(provider, dict):
        provider = {}

    existing_models = provider.get("models")
    existing_by_id: dict[str, dict[str, object]] = {}
    if isinstance(existing_models, list):
        for item in existing_models:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                existing_by_id[item["id"]] = item

    window, output = _context_window(ctx), _max_output_tokens(ctx)
    new_models: list[dict[str, object]] = []
    for model_id in _ordered_model_ids(ctx):
        entry = _openclaw_model_entry(model_id, window, output)
        for key, value in existing_by_id.get(model_id, {}).items():
            entry.setdefault(key, value)
        new_models.append(entry)

    provider["baseUrl"] = ctx.server.openai_base_url
    provider["apiKey"] = "freetoken-local"
    provider["api"] = "openai-completions"
    provider["models"] = new_models
    providers[OPENCLAW_PROVIDER] = provider
    models_section["providers"] = providers
    config["models"] = models_section

    agents = _dict_child(config, "agents")
    defaults = _dict_child(agents, "defaults")
    model_config = _dict_child(defaults, "model")
    model_config["primary"] = f"{OPENCLAW_PROVIDER}/{ctx.model.model_id}"
    defaults["model"] = model_config
    agents["defaults"] = defaults
    config["agents"] = agents
    return config


def _clear_openclaw_session_overrides(primary_model: str) -> None:
    path = _openclaw_sessions_path()
    if not path.exists():
        return
    sessions = _load_json_object(path)
    changed = False
    for session in sessions.values():
        if not isinstance(session, dict):
            continue
        if "modelOverride" in session:
            del session["modelOverride"]
            changed = True
        if "providerOverride" in session:
            del session["providerOverride"]
            changed = True
        model = session.get("model")
        if isinstance(model, str) and model and model != primary_model:
            session["model"] = primary_model
            changed = True
    if changed:
        _write_json_with_backup(path, sessions)


def prepare_openclaw(ctx: LaunchContext) -> CommandSpec:
    if not ctx.dry_run:
        config_path = _openclaw_config_path()
        config = _load_json_object(config_path)
        if not _openclaw_has_freetoken_provider(config):
            if not ctx.assume_yes and not _confirm_openclaw_first_patch(ctx, config_path):
                raise RuntimeError("OpenClaw config patch cancelled")
        _write_json_with_backup(config_path, _patch_openclaw_config(ctx, config))
        _clear_openclaw_session_overrides(ctx.model.model_id)

    argv = ["openclaw", *ctx.extra_args] if ctx.extra_args else ["openclaw", "chat"]
    return CommandSpec(argv=argv, env={}, unset_env=OPENCLAW_CLEAR_ENV)


def _hermes_config_path() -> Path:
    return Path.home() / ".hermes" / "config.yaml"


def prepare_hermes(ctx: LaunchContext) -> CommandSpec:
    """Nous Research's Hermes agent (hermes-agent.nousresearch.com). Its config.yaml `model:`
    section takes a custom OpenAI-compatible endpoint exactly like its Ollama/vLLM setup — when
    ``base_url`` is set Hermes bypasses the named provider and calls that endpoint directly. We
    patch only the model section (preserving the user's other config) to point at FreeToken.
    `max_tokens` stays unset: Hermes budgets compaction from context_length alone."""
    if not ctx.dry_run:
        import yaml  # ships with transformers (a core dependency); only needed on this path

        config_path = _hermes_config_path()
        config: dict[str, object] = {}
        if config_path.exists():
            try:
                loaded = yaml.safe_load(config_path.read_text())
            except yaml.YAMLError:
                loaded = None
            if isinstance(loaded, dict):
                config = loaded

        model_section = config.get("model")
        if not isinstance(model_section, dict):
            model_section = {}
        model_section["default"] = ctx.model.model_id
        model_section["provider"] = "custom"
        model_section["base_url"] = ctx.server.openai_base_url
        # A non-empty dummy: FreeToken is unauthenticated, but some clients reject an empty key.
        model_section["api_key"] = HERMES_API_KEY
        # Hermes' name for the window (prompt + generation). Left unset it auto-detects, and a
        # miss lands on a 32k default that trips the floor below.
        window = _context_window(ctx)
        model_section["context_length"] = window
        config["model"] = model_section

        if window < HERMES_MIN_CONTEXT_LENGTH:
            print(
                f"warning: Hermes refuses to start under {HERMES_MIN_CONTEXT_LENGTH} tokens of "
                f"context and this model reports {window}.",
                file=sys.stderr,
            )

        _write_text_with_backup(config_path, yaml.safe_dump(config, sort_keys=False))

    argv = ["hermes", *ctx.extra_args] if ctx.extra_args else ["hermes", "chat"]
    return CommandSpec(argv=argv, env={})


def _dsh_home() -> Path:
    home = os.environ.get("DSH_HOME")
    return Path(home).expanduser() if home else Path.home() / ".dsh"


def _warn_dsh_node_version() -> None:
    node = shutil.which("node")
    if node is None:
        return
    try:
        raw = subprocess.run(
            [node, "--version"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        parts = raw.lstrip("v").split(".")
        version = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return
    if version < DSH_MIN_NODE:
        floor = ".".join(str(part) for part in DSH_MIN_NODE)
        print(
            f"warning: dsh requires Node.js >={floor} and node reports {raw}.",
            file=sys.stderr,
        )


def prepare_dsh(ctx: LaunchContext) -> CommandSpec:
    """DeepSeek Harness (dsh). Like the codex profile, the wiring lives in its
    own settings document: a ``--patch`` overlay repoints dsh's settings row at
    ``freetoken-launch.settings.yaml`` for this invocation only, so the user's
    ``settings.yaml`` is never read or written. The llm-deepseek adapter (route
    ``deepseek-official``) is used rather than a custom llm-pi-ai provider: it
    replays tool-call arguments byte-verbatim and reasoning as
    ``reasoning_content``, while llm-pi-ai's JSON round-trip can change argument
    values and key order — drift that survives render canonicalization, cuts
    the prefix cache, and shows the model a rewrite of its own output. The
    dummy DEEPSEEK_API_KEY satisfies dsh's non-empty key requirement; FreeToken
    itself is unauthenticated."""
    settings_path = _dsh_home() / DSH_LAUNCH_SETTINGS_NAME
    patch_path = _dsh_home() / DSH_LAUNCH_PATCH_NAME
    if not ctx.dry_run:
        import yaml  # ships with transformers (a core dependency); only needed on this path

        _warn_dsh_node_version()
        # Preserve state dsh itself writes back into the active settings doc
        # (model re-picks, onboarding acks); only our two sections are re-pinned.
        config: dict[str, object] = {}
        if settings_path.exists():
            try:
                loaded = yaml.safe_load(settings_path.read_text())
            except yaml.YAMLError:
                loaded = None
            if isinstance(loaded, dict):
                config = loaded

        window, output = _context_window(ctx), _max_output_tokens(ctx)
        config["llm-deepseek"] = {
            "baseURL": ctx.server.openai_base_url,
            "models": [
                {
                    "id": model_id,
                    "name": f"{model_id} (FreeToken)",
                    "contextWindow": window,
                    "maxTokens": output,
                }
                for model_id in _ordered_model_ids(ctx)
            ],
        }
        config["agent-default-model"] = {
            "provider": "deepseek-official",
            "model": ctx.model.model_id,
        }
        _write_text_with_backup(settings_path, yaml.safe_dump(config, sort_keys=False))

        patch = [{"id": "settings", "config": {"path": str(settings_path)}}]
        _write_text_with_backup(patch_path, yaml.safe_dump(patch, sort_keys=False))

    # The explicit --profile form (equivalent to the `web` alias) lets the
    # normalization below swap the profile from extra args.
    profile, app_args = "web", list(ctx.extra_args)
    if app_args and app_args[0] == "web":
        app_args = app_args[1:]
    elif len(app_args) >= 2 and app_args[0] == "--profile":
        profile, app_args = app_args[1], app_args[2:]
    argv = ["dsh", "--profile", profile, "--patch", str(patch_path), *app_args]
    return CommandSpec(
        argv=argv,
        env={
            "DEEPSEEK_BASE_URL": ctx.server.openai_base_url,
            "DEEPSEEK_API_KEY": DSH_API_KEY,
            "DSH_TELEMETRY_DISABLED": "1",
        },
        unset_env=CLOUD_PROVIDER_API_KEY_ENV,
    )


PREPARERS = {
    "codex": prepare_codex,
    "claude": prepare_claude,
    "opencode": prepare_opencode,
    "openclaw": prepare_openclaw,
    "hermes": prepare_hermes,
    "dsh": prepare_dsh,
}


def parse_argv(argv: list[str], prog: str = "python -m freetoken.launch") -> argparse.Namespace:
    if "--" in argv:
        separator = argv.index("--")
        launch_args = argv[:separator]
        extra_args = argv[separator + 1 :]
    else:
        launch_args = argv
        extra_args = []

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Configure and launch a coding agent against an existing FreeToken server.",
    )
    parser.add_argument("agent", choices=sorted(PREPARERS))
    parser.add_argument("--server", help="FreeToken server origin or /v1 URL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes and command",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically approve install and config prompts",
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Run the agent installer even when the CLI is already installed",
    )
    parser.add_argument(
        "--config",
        dest="config_only",
        action="store_true",
        help="Install and configure the agent without launching it",
    )
    parser.add_argument(
        "--install-only",
        dest="install_only",
        action="store_true",
        help="Install the agent CLI if missing, then exit — needs no server and no loaded model",
    )
    args = parser.parse_args(launch_args)
    args.extra_args = extra_args
    return args


def print_dry_run(ctx: LaunchContext, spec: CommandSpec) -> None:
    print(f"FreeToken server: {ctx.server.origin}")
    print(f"OpenAI base URL: {ctx.server.openai_base_url}")
    print(f"Model: {ctx.model.model_id}")
    print(f"Context window: {_context_window(ctx)}")
    print(f"Command: {shlex.join(spec.argv)}")
    if spec.env:
        print("Environment:")
        for key, value in sorted(spec.env.items()):
            print(f"  {key}={value}")
    if spec.unset_env:
        print("Unset environment:")
        for key in spec.unset_env:
            print(f"  {key}")


def _agent_binary_filename(binary: str) -> str:
    if os.name == "nt" and not binary.endswith(".exe"):
        return f"{binary}.exe"
    return binary


def _agent_binary_fallbacks(agent: str, binary: str) -> list[Path]:
    home = Path.home()
    filename = _agent_binary_filename(binary)
    if agent == "codex":
        return [home / ".local" / "bin" / filename]
    if agent == "claude":
        return [
            home / ".local" / "bin" / filename,
            home / ".claude" / "local" / filename,
        ]
    if agent == "opencode":
        return [home / ".opencode" / "bin" / filename]
    if agent == "openclaw":
        return [home / ".npm-global" / "bin" / filename]
    if agent == "hermes":
        return [home / ".local" / "bin" / filename]
    if agent == "dsh":
        return [home / ".npm-global" / "bin" / filename]
    return []


def resolve_agent_binary(agent: str, binary: str) -> str | None:
    names = [binary]
    if agent == "openclaw" and binary == "openclaw":
        names.append("clawdbot")
    for name in names:
        path = shutil.which(name)
        if path is not None:
            return path
    for path in _agent_binary_fallbacks(agent, binary):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def _confirm_install(installer: AgentInstaller, force_reinstall: bool = False) -> bool:
    if force_reinstall:
        print(f"{installer.display_name} is already installed.", file=sys.stderr)
        action = "Reinstall"
    else:
        print(f"{installer.display_name} is not installed.", file=sys.stderr)
        action = "Install"
    print(f"Install command: {shlex.join(installer.argv)}", file=sys.stderr)
    if not sys.stdin.isatty():
        print("Run this command in an interactive terminal to install it.", file=sys.stderr)
        return False
    while True:
        answer = input(f"{action} {installer.display_name} now? [y/n] ").strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False
        print("Please answer y or n.", file=sys.stderr)


def ensure_agent_installed(
    agent: str,
    binary: str,
    assume_yes: bool = False,
    force_reinstall: bool = False,
) -> str:
    if not force_reinstall and (path := resolve_agent_binary(agent, binary)):
        return path

    installer = AGENT_INSTALLERS.get(agent)
    if installer is None:
        raise RuntimeError(f"{binary} was not found on PATH")

    missing = [dep for dep in installer.dependencies if shutil.which(dep) is None]
    if missing:
        joined = "\n  ".join(missing)
        raise RuntimeError(
            f"{installer.display_name} is not installed and required dependencies are missing\n\n"
            f"Install the following first:\n  {joined}\n\n"
            f"Then re-run:\n  ft launch {agent}"
        )

    if not assume_yes:
        confirmed = (
            _confirm_install(installer, force_reinstall=True)
            if force_reinstall
            else _confirm_install(installer)
        )
        if not confirmed:
            raise RuntimeError(f"{agent} installation cancelled")

    verb = "Reinstalling" if force_reinstall else "Installing"
    print(f"\n{verb} {installer.display_name}...", file=sys.stderr)
    env = os.environ.copy()
    env.update(dict(installer.env))
    try:
        subprocess.run(list(installer.argv), check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to install {agent}: {exc}") from exc

    path = resolve_agent_binary(agent, binary)
    if path is None:
        raise RuntimeError(
            f"{binary} was installed but was not found\n\n"
            "You may need to restart your shell"
        )

    print(f"{installer.display_name} installed successfully\n", file=sys.stderr)
    return path


def run_command(spec: CommandSpec) -> int:
    env = os.environ.copy()
    for key in spec.unset_env:
        env.pop(key, None)
    env.update(spec.env)
    return subprocess.run(spec.argv, env=env).returncode


def main(argv: list[str] | None = None, prog: str = "python -m freetoken.launch") -> int:
    args = parse_argv(sys.argv[1:] if argv is None else argv, prog=prog)
    try:
        # Installing a CLI needs neither a server nor a loaded model. Every preparer's argv[0] is
        # the agent id (``resolve_agent_binary`` handles the openclaw→clawdbot alias), so the
        # binary name is known without building a spec — which is what let this jump ahead of
        # ``resolve_server_url``. Without it, the desktop's "Install" action would force the user
        # to load a model just to install an agent.
        if args.install_only:
            ensure_agent_installed(
                args.agent,
                args.agent,
                assume_yes=args.yes,
                force_reinstall=args.force_reinstall,
            )
            return 0

        server = resolve_server_url(args.server)
        model = discover_server_model(server)
        plan_ctx = LaunchContext(
            server=server,
            model=model,
            extra_args=args.extra_args,
            dry_run=True,
            assume_yes=args.yes,
        )
        spec = PREPARERS[args.agent](plan_ctx)
        if args.dry_run:
            print_dry_run(plan_ctx, spec)
            return 0
        binary_path = ensure_agent_installed(
            args.agent,
            spec.argv[0],
            assume_yes=args.yes,
            force_reinstall=args.force_reinstall,
        )
        ctx = LaunchContext(
            server=server,
            model=model,
            extra_args=args.extra_args,
            dry_run=False,
            assume_yes=args.yes,
        )
        spec = PREPARERS[args.agent](ctx)
        if binary_path != spec.argv[0]:
            spec = CommandSpec(
                argv=[binary_path, *spec.argv[1:]],
                env=spec.env,
                unset_env=spec.unset_env,
            )
        if args.config_only:
            return 0
        return run_command(spec)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
