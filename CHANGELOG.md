# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Security

- A control character that Python treats as whitespace (vertical tab, form feed, U+001C to U+001F, NEL) inside
  a tool-call argument value was rendered as a space. Inside a word it split the word in two, so
  `del<VT>ete` did not match a `delete` keyword and the call was allowed. A value that contains control
  characters is now also rendered with them removed and with them as spaces, and rules are matched against
  every rendering. This also applies to identifier values such as `skip<NUL>_audit`.
- **Behaviour change:** a value with more than one of these whitespace control characters is denied, because
  each one could be a separator or sit inside a word. Pass your own `tool_call_formatter` if your tools
  legitimately receive such values.
- Checking whether an argument value is a single identifier no longer takes exponential time on crafted
  values.

Not covered:

- A value with several non-whitespace control characters (for example NUL or ESC) where one sits inside a
  word and another separates words, such as `del<NUL>ete<SOH>everything`. Such values are not denied, because
  text with terminal colour codes contains many of these characters. The printable part of such a code, for
  example `[0m`, also stays in the text.
- Other characters that Python treats as whitespace inside a word, such as no-break space or U+2028.
- Invisible format characters such as zero-width space, which acgs-lite 3.x does not remove.
- Control characters in tool names and argument names, and values whose `str()` escapes control characters
  (for example `bytes`).

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
