# MiMo Template Contract

`mimo_chat_contract.jinja` is an authored, text-only regression fixture under the
repository's Apache-2.0 contribution license. It is not an upstream template or a
copy of the supplied checkpoint artifact. It implements only the interfaces needed
by the generation-position matrix and assistant tool-history round trip: text
messages, tool definitions, mapped function arguments, explicit assistant reasoning,
and the generation suffix. It does not cover multimodal/custom-tool serialization,
token IDs, checkpoint loading, or full-model behavior.

The enforced SHA256 of the authored fixture, after CRLF-to-LF normalization for
ordinary Windows checkouts, is
`5d1656529ef121e333c5a5c65aa1717f8ac7a73ac48c801968fe9c6f354ca9c4`.
Default tests need no external checkpoint files and never skip for a missing template.
They render default/on/off thinking with and without tools, invoke production
`_make_reasoning_parser`, check its initial non-reasoning state against the suffix,
and check plain text, explicit reasoning, and tool envelopes in whole/chunked output.
The history test checks arguments parsed through the OpenAI API, converted back to
template messages, and serialized verbatim by the fixture.

## External Corroboration

The user supplied `validation/real_ckpt_ref/chat_template.jinja` outside this worktree
at `C:\DockerContainers\free-token-mimo-2.6-flash`. Its raw-byte SHA256 was independently
recomputed during this fix and is
`853650bee57bf95020373e4c928bd5a4b41b9915adf964a77711d2b49a291887`.
This also matches the historical `validation/final-fix-report.md`, which is not
current execution evidence. The supplied artifact renders a bare assistant header
by default/when thinking is on, and an already-closed empty think block when off,
irrespective of tool presence. Those are the contract boundaries tested here.

No authenticated upstream repository/revision or redistribution license for that
template was established from the supplied local files, so it is not vendored.
The adjacent config's `conversion_metadata.source_revision`
`3b38d063180c3e4aed9691fdc735f3d10b266ee4` is only a conversion metadata claim;
it does not authenticate the template or tokenizer revision. No upstream revision
is asserted for the authored fixture either.

Set `FREETOKEN_MIMO_CHAT_TEMPLATE` to that external artifact to run the same matrix
and history test against it instead of the authored fixture. The tracked fixture's
pin is still checked first. Missing files or different raw-byte hashes fail, never
skip or silently accept an alternate template. This corroborates only the supplied
bytes, not arbitrary MiMo checkpoints. A different deployed template needs its own
verified contract; the production factory does not dynamically inspect templates.

```bash
python -m pytest tests/server/test_openai_api.py -k 'mimo and (pinned_template or template_pin)' -q -o addopts= -p no:cacheprovider -rs
```
