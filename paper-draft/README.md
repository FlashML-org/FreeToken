# Technical white paper release candidate package

`amd_strix_halo_freetoken_port_draft.md` is a research-paper draft based only on the repository's recorded evaluated-system evidence, a live hardware and software manifest captured on 30 August 2026, and the supplied FreeToken paper.

`paper.tex` and `references.bib` are the venue-neutral LaTeX source. Build them with a standard TeX distribution using `pdflatex paper`, `bibtex paper`, then `pdflatex paper` twice. Select the final venue template only after the submission path is fixed; the existing LaTeX source deliberately avoids venue-specific formatting.

For reviewer circulation, `output/pdf/freetoken-amd-strix-halo-white-paper-v0.1.0-rc1.pdf` is a polished release-candidate copy generated directly from the Markdown manuscript. Rebuild it with the bundled workspace Python runtime and `scripts/build_paper_pdf.py`. The renderer is intentionally venue-neutral; the Markdown and LaTeX manuscripts remain the authoritative editable sources.

It is intentionally written as a systems-port and controlled-evaluation paper, not as a claimed replication of FreeToken's published NVIDIA results. The manuscript now includes the full non-sensitive evaluated-system platform table and an artifact-availability section. Before submission, convert the plain references to the target venue's BibTeX style, attach the raw artifact bundle listed in the paper's reproducibility section, and create a tagged archival release.

The immediate evidence gaps are: the upstream Qwen benchmark contract, five-sample repetitions for the Q4 control, an expanded quality suite, long-context and agentic workloads, parameterized public model-launch recipes, and a second clean-host matrix.

Use [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md) for the final tag, Zenodo archive, and DOI substitution steps. The release notes in [RELEASE_NOTES_v0.1.0-rc1.md](RELEASE_NOTES_v0.1.0-rc1.md) are the proposed GitHub release body.

## Contact

For manuscript correspondence, replication questions, or technical collaboration, contact David Bourdeau at [davidbourdeau@gmail.com](mailto:davidbourdeau@gmail.com). The project repository is <https://github.com/dbourdea/FreeToken>. Enable and link the public issue tracker before release so reproducible software defects and proposed changes have a searchable public route.
