# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## 0.1.0 (unreleased)

### Added

- `ACGSMiddleware`, a LangChain `AgentMiddleware` that checks each tool call against an ACGS constitution in
  `wrap_tool_call` and `awrap_tool_call`.
- Review verdicts pause the agent with a LangGraph `interrupt` (`ACGSReviewRequest`, same keys as
  `ToolApprovalRequest`); `approve`, `reject` and `edit` answers, with edited arguments evaluated again.
- `format_tool_call`, the documented text representation of a tool call that ACGS rules are evaluated against.
- Fail-closed handling: evaluation errors and invalid review answers deny the call.
