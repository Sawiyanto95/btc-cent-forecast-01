# BTC 4-hour forecast — make.fun bounty `btc-cent-forecast-01`

Forecast Bitcoin's price 4 hours ahead, 12 times over 48 hours.
Entry for make.fun: a program that reads BTC 1-minute closes for the
72 hours up to a checkpoint and guesses the price 240 minutes later.
Score is mean absolute error in cents; it must beat the persistence
baseline (guess = last close) to win any prize.

## Files

- `main.py` — the entry. Python 3.13, **standard library only**, no
  network, deterministic.
- `test_forecast.py` — test harness (synthetic + real-data backtest).
- `TEST_REPORT.md` — test results.
- `LICENSE` — MIT.

## How it works

Work in log-price space. Inside each 72 h case, build walk-forward
training examples: for every minute `i`, the target is the 240-minute
log return and the features (all from data ≤ `i`, no leakage) are
lagged log returns (5/15/60/240/480/1440 min), gaps to the 4 h / 8 h /
24 h / 72 h means, recent 4 h volatility, and the 6 h trend slope
projected 240 min ahead.

A small ridge regression (closed-form, pure Python, 12 features) is
fitted with an intercept. The ridge penalty and a forecast scale
`alpha` are chosen jointly by chronological walk-forward validation
(train first 40%, validate next 25%, with a 240-minute gap between
them — the gap matters, because 240-minute target windows overlap).
`alpha = 0` is exactly persistence, so a ridge that does not beat
persistence in validation by at least 3% is switched off. The
forecast is shrunk toward 0 in proportion to recent volatility and
clamped to ±0.8% total 4 h move.

Fallbacks: fewer than 300 prices → last price. Ridge failure →
persistence plus a tiny mean-reversion term.

## How to run locally (grader-style)

```bash
mkdir -p /tmp/in /tmp/out
# put one case in /tmp/in/<id>.in, e.g. one price per line
python3 -P -s main.py   # with env vars, see below
```

The grader runs it as `python3 -P -s /bot/main.py` with
`MAKEFUN_INPUT_DIR=/input`, `MAKEFUN_OUTPUT_DIR=/output`,
`MAKEFUN_BOT_DIR=/bot`. It reads `/input/<id>.in` and writes
`/output/<id>.out` as JSON:
`{"forecasts": [{"case": "<id>", "price": <guess>}]}`.
A plain number is an invalid case (scored as persistence, entry
takes no prize), so the output format matters as much as the
forecast.

Locally, the same layout works:

```bash
MAKEFUN_INPUT_DIR=/tmp/in MAKEFUN_OUTPUT_DIR=/tmp/out python3 -P -s main.py
cat /tmp/out/*.out
```

Practice cases are in `kit/btc-cent-forecast-01-kit/input/` — run
`main.py` on them the same way to reproduce the practice score in
`TEST_REPORT.md` (43,557 cents MAE, vs 47,055 for persistence).

## Input / output format

Exact grader format (confirmed from the kit's `competition.yaml`):
input is one JSON object per `.in`,
`{"case": <id>, "markets": [{"id": "btc-usdc", "prices": [4320 closes]}]}`,
last price = checkpoint close. Output is the JSON `forecasts`
object above, price rounded to 5 decimals and clamped to
(0, 10,000,000). If a case file is not valid JSON, `main.py` falls
back to scanning it for numbers and guessing the last one
(persistence) so the output is still validly shaped instead of
crashing the run.
