from unittest.mock import Mock


def test_load_tokenizer_allows_checkpoint_custom_code(monkeypatch):
    from freetoken.utils import hf

    tokenizer = Mock(chat_template="template")
    load = Mock(return_value=tokenizer)
    monkeypatch.setattr(hf.AutoTokenizer, "from_pretrained", load)
    monkeypatch.setattr(hf, "hf_hub_download", Mock())

    assert hf.load_tokenizer("example/custom-tokenizer") is tokenizer
    load.assert_called_once_with(
        "example/custom-tokenizer", trust_remote_code=True
    )
