from freetoken.models.gguf import tokenizer as gguf_tokenizer


def test_gemma_gpt2_gguf_uses_raw_vocab_merges(monkeypatch):
    from transformers.integrations import gguf

    tokens = ["<pad>", "<eos>", "<bos>", "<unk>", "\n", "\n\n", "▁", "a", "▁a"]
    token_types = [3, 3, 3, 3, 1, 1, 1, 1, 1]
    metadata = {
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": token_types,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.merges": ["\n \n", "▁ a"],
        "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.unknown_token_id": 3,
        "tokenizer.ggml.padding_token_id": 0,
    }
    converter_metadata = {
        "tokenizer_type": "gpt2",
        "tokens": tokens,
        "token_type": token_types,
        "merges": metadata["tokenizer.ggml.merges"],
    }
    monkeypatch.setattr(
        gguf,
        "get_gguf_tokenizer",
        lambda _: ("gemma4", converter_metadata, {}),
    )
    monkeypatch.setattr(gguf_tokenizer, "load_gguf_metadata", lambda _: metadata)
    monkeypatch.setattr(gguf_tokenizer, "gguf_architecture", lambda _: "gemma4")

    tokenizer = gguf_tokenizer.load_gguf_tokenizer("model.gguf")

    text = "a\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text


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


def test_qwen_image_token_id_matches_reconstructed_vocabulary():
    meta = {
        "tokenizer.ggml.tokens": ["<|endoftext|>", "hello"],
        "tokenizer.ggml.model": "gpt2",
    }

    assert gguf_tokenizer.qwen_added_token_id(meta, "<|image_pad|>") == 13
