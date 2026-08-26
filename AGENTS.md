# Repository Development Rules

이 문서의 규칙은 repository 전체에 적용된다.

- Do not create version-suffixed implementation files such as `v2`, `v3`, or `final`.
- Prefer modifying canonical modules and changing YAML config.
- Before creating a new Python file, check whether an existing module can be reused.
- One responsibility should have one canonical implementation.
- Do not create a new runner for each deployment experiment.
- Do not commit generated datasets or arbitrary experiment outputs.
- Keep generated model conversions, firmware builds, and reports outside source directories.
- Keep source model, converted model, firmware, and runtime output provenance explicit and separate.
- Do not put research training logic, dataset generation, or architecture exploration in this E84 repository.
- Do not put quantization, firmware, or HIL logic in the research repository.
- Keep README `Current Status` synchronized with the actual project state; it is the canonical status.
- Keep the complete pipeline understandable within five minutes to a developer learning the codebase.
- Prefer explicit, simple code over clever abstractions.
- Do not introduce frameworks, packages, or abstractions before they are needed.
- Delete unused code instead of keeping `old`, `backup`, or versioned copies; Git history is the archive.
- Do not silently copy code, datasets, models, outputs, firmware, or dependencies from the legacy `/d/shin/Infineon` repository.
- Any future migration from the legacy repository must be explicit and reviewed.
- Accept only frozen research artifacts with a reviewed contract, provenance, metrics, and checksum.
