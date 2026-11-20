# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## 0.2.0 (2026-11-20)

### Added

- Offline `examples/deny_and_audit.py`: one allowed and one denied LangChain tool call, with an application
  decision trail.
- Regression coverage for nested JSON-native arguments and bytes/bytearray formatting against acgs-lite 4.0.0.

### Changed

- Bump the adapter version to 0.2.0 and document the compatibility matrix: acgs-lite 4.0.0 is tested in the
  default environment; the CI `test-min` job keeps testing the 3.0.1 floor. Runtime support remains
  `acgs-lite>=3.0.1,<5` with unchanged dependency constraints.
- Clarify that the adapter uses text-based rule validation and does not issue signed receipts.

## 0.1.1 (2026-10-20)

### Fixed

- When a blocking rule and a review rule both match a tool call, the denial reason now lists only the blocking
  rules after "Blocked by" and names the matched review rules separately ("Review rules that also matched: ...").
  Previously the review rule was also listed under "Blocked by". The verdict was already `deny` and is unchanged;
  `rule_ids` still contains every matched rule.

## 0.1.0 (2026-10-20)

### Added

- `ACGSMiddleware`, a LangChain `AgentMiddleware` that checks each tool call against an ACGS constitution in
  `wrap_tool_call` and `awrap_tool_call`.
- Review verdicts pause the agent with a LangGraph `interrupt` (`ACGSReviewRequest`, same keys as
  `ToolApprovalRequest`); `approve`, `reject` and `edit` answers, with edited arguments evaluated again.
- `format_tool_call`, the documented text representation of a tool call that ACGS rules are evaluated against.
- Fail-closed handling: evaluation errors and invalid review answers deny the call.
