# CSC advisory latency observations

Measured locally on 2026-11-10 using the CSC Aitta API and the existing
standard-mode advisory prompt. No credentials are included in this report.

## Method

- Same first confirmed marine alert from the committed voyage artifacts.
- Two sequential, uncached requests per model, with temperature 0.2 and
  max_tokens 2048. GPT-OSS additionally used reasoning_effort "low".
- GPT-OSS elapsed times include HTTP and response handling, excluding
  retrieval. Qwen app times include HTTP and advisory validation, excluding
  retrieval; timeouts produced deterministic offline advisories.
- Requests used a 30-second connect/read inactivity timeout, not a hard
  end-to-end deadline. No automatic retries.

## Observations

| Model | Attempt 1 | Attempt 2 | Result |
| --- | --- | --- | --- |
| Qwen/Qwen3.6-35B-A3B | 30.07 s | 30.04 s | Both timed out; offline fallback |
| openai/gpt-oss-120b, low reasoning | 31.05 s | 30.04 s | Both timed out |

A subsequent authenticated worker-status check for GPT-OSS reported
`requested`, not `running`. Its measurements therefore do not establish
warmed-up generation latency or show whether it is faster than Qwen.

For Qwen, a longer diagnostic request returned HTTP 200 in 33.81 seconds,
but hit the 2048-token limit with reasoning present and no final advisory
content. The worker had reported `running` before that diagnostic.
A further diagnostic with chat_template_kwargs.enable_thinking=false
returned in 34.60 seconds with the same length-limit/no-final-content
outcome. That flag did not resolve the issue on this endpoint in this test.

These small, sequential samples are not a benchmark or an estimate of
typical latency. No successful GPT-OSS generation was measured. Repeat
with a running worker before making performance claims.

## App behavior

The selected default is openai/gpt-oss-120b with low reasoning effort.
Unavailable or invalid LLM responses fall back to the deterministic
composer with a safe warning. The advisory footer reports the original
attempt duration; cached results reuse that duration for up to five
minutes. Predictions, intervals and alert decisions remain independent
of the LLM.

Local raw evidence is retained outside the repository as
`~/ua_qwen_latency.json` and `~/ua_gpt_oss_latency.json`.
