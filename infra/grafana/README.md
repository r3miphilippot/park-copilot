# Monitoring with Grafana Cloud

The API exposes its metrics in the Prometheus text format at `GET /metrics/prometheus`
(protected by `METRICS_TOKEN`). Grafana Cloud's free tier scrapes that URL and
[`park-copilot-dashboard.json`](park-copilot-dashboard.json) turns it into a dashboard.

| Metric | Type | Labels |
|---|---|---|
| `park_copilot_chat_requests_total` | counter | `status`: ok, error, rate_limited, daily_cap |
| `park_copilot_chat_duration_seconds` | histogram | - |
| `park_copilot_llm_fallbacks_total` | counter | - |
| `park_copilot_tool_calls_total` | counter | `tool`, `ok` |
| `park_copilot_tool_duration_seconds` | histogram | `tool` |

Counters and histograms rather than precomputed percentiles: Grafana computes rates and p95 over
any time window, and `rate()` handles counters restarting at 0 after a redeploy.

## Setup

1. Generate a token, e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"`, and
   add it as `METRICS_TOKEN` to the Render environment.
2. Create a free [Grafana Cloud](https://grafana.com/products/cloud/) account.
3. *Connections → Add new connection → Metrics Endpoint*: URL
   `https://park-copilot.onrender.com/metrics/prometheus`, authentication *Bearer* with the
   token (or *Basic* with any username and the token as password), scrape interval 1 minute.
4. *Dashboards → New → Import*: upload `park-copilot-dashboard.json` and pick the Grafana Cloud
   Prometheus data source.
5. Optional alert: *error rate > 5 %* or *latency p95 > 15 s* for 10 minutes.

A scrape every minute also keeps the free Render instance awake.
