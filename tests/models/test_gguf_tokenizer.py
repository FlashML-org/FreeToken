from freetoken.models.gguf import tokenizer as gguf_tokenizer


def test_gguf_eos_token_ids_uses_metadata_and_gemma_turn_tokens(monkeypatch):
    meta = {
        "tokenizer.ggml.tokens": ["</s>", "<|tool_response>", "<turn|>"],
        "tokenizer.ggml.eos_token_id": 0,
    }
    monkeypatch.setattr(gguf_tokenizer, "load_gguf_metadata", lambda _: meta)
    monkeypatch.setattr(gguf_tokenizer, "gguf_architecture", lambda _: "gemma4")

    assert gguf_tokenizer.gguf_eos_token_ids("model.gguf") == {1, 2}


def test_gguf_eos_token_ids_includes_reconstructed_qwen_tokens(monkeypatch):
    meta = {
        "tokenizer.ggml.tokens": ["!", "hello"],
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.eos_token_id": 1,
    }
    monkeypatch.setattr(gguf_tokenizer, "load_gguf_metadata", lambda _: meta)
    monkeypatch.setattr(gguf_tokenizer, "gguf_architecture", lambda _: "qwen3_5_moe")

    assert gguf_tokenizer.gguf_eos_token_ids("model.gguf") == {1, 2, 4}


def test_gguf_eos_token_ids_accounts_for_existing_qwen_added_tokens(monkeypatch):
    meta = {
        "tokenizer.ggml.tokens": ["<|endoftext|>", "hello"],
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.eos_token_id": 1,
    }
    monkeypatch.setattr(gguf_tokenizer, "load_gguf_metadata", lambda _: meta)
    monkeypatch.setattr(gguf_tokenizer, "gguf_architecture", lambda _: "qwen3_5_moe")

    assert gguf_tokenizer.gguf_eos_token_ids("model.gguf") == {0, 1, 3}
