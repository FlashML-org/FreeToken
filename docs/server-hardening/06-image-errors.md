# Consistent text-only image rejection

Every recognized image block now raises the same shared
GenerationError("image content not supported by this text-only server").
Chat content parts, Anthropic messages/count_tokens (including system content and
nested tool results), and Responses input content return a protocol-shaped HTTP
400 before admission. Mixed text/image prompts fail as a whole rather than having
the image silently removed. Existing text, tools, thinking and unknown opaque
Anthropic block behavior is preserved.

vLLM, SGLang and TensorRT-LLM support multimodal input only through model/processor
and engine paths that can consume it; their serving APIs cannot turn an image into
a text-only model capability. FreeToken's shared generation normalization is the
right boundary to enforce its online server's current text-only contract. The
offline engine's separate multimodal facilities are unchanged. This explicitly
requested rejection replaces Anthropic's previous silent image drop; it does not
introduce a multimodal backend.

Validation: protocol-level 400s for mixed and image-only requests, both streaming
and buffered, plus nested image rejection, shared exception identity/message and
unchanged text normalization. No GPU/model inference is needed for this boundary.
