# Test report — btc-cent-forecast-01

Run: `python3 test_forecast.py` in `~/workspace/btc-forecast/`
(last full run: 2026-11-15, Python 3.12.3 locally; entry targets
Python 3.13, stdlib only). All MAE figures are in cents
(100 × mean |guess − actual| in dollars). "Beats persistence" means
the model MAE is lower than guessing the last close.

## Headline: kit practice cases (the real grader format)

The kit's sealed practice instance (`kit/btc-cent-forecast-01-kit/input/`,
cbBTC/USDC Orca prices, checkpoints 2026-11-10 06:00 → 2026-11-11 02:00 UTC).
Actual for p00–p04 is the next case's checkpoint close (its target
minute); p05's target (2026-11-11 06:00) is past the practice window,
so it is scored for output validity only, not MAE.

| case | last | actual | guess | err model | err persist |
|------|------|--------|-------|-----------|-------------|
| p00 | 74764.92 | 74447.80 | 74694.25 | 24,645 | 31,711 |
| p01 | 74447.80 | 73680.84 | 74447.80 | 76,697 | 76,697 |
| p02 | 73680.84 | 73210.50 | 73680.84 | 47,033 | 47,033 |
| p03 | 73210.50 | 73777.35 | 73257.73 | 51,962 | 56,685 |
| p04 | 73777.35 | 73545.88 | 73720.38 | 17,450 | 23,147 |

**Practice MAE (p00–p04): model 43,557.4 cents, persistence 47,054.7 cents
— beats persistence by 7.4%.** For comparison, the earlier heuristic
blend scored 46,812.5 on the same cases (−0.5%). p01/p02 fall back
to persistence (alpha = 0): validation inside those cases did not
trust the ridge, which is the intended safe behaviour.

## Real BTC data (Binance US public klines, no key)

5 cases, 72 h each, 4 h apart, from the most recent ~6,000 minutes.
The window shifts as the live price moves, so the numbers move
slightly between runs; the model beat persistence in each of the
last three runs.

- Latest run: model **83,993** vs persistence **84,798** cents (−1.0%)
- Earlier runs: 80,132 vs 80,635 (−0.6%); 85,082 vs 85,887 (−0.9%)
- Larger backtest (20 cases, ~10 days, 15,000 minutes, price falling
  $80,146 → $64,192): model **67,296** vs persistence **69,470** (−3.1%)

Note: Binance.com is geo-blocked from this machine ("Service
unavailable from a restricted location"), so Binance US
(`api.binance.us`) was used. The kit practice prices are cbBTC/USDC
from an Orca pool, not Binance — the practice score above is the
one that matches the grader's data source.

## Synthetic series (5 checkpoints each, 4 h apart)

Deterministic series generated with `random.Random(42)`; 1-min vol
0.08%, start $64,000.

| series | model MAE | persistence MAE | verdict |
|--------|-----------|-----------------|---------|
| random walk | 55,461 | 52,054 | +6.5% worse |
| trending (+1.7%/day drift) | 49,701 | 50,926 | **−2.4% beats** |
| mean-reverting (OU, 6 h half-life) | 51,478 | 50,541 | +1.9% worse |
| flat (vol 0.005%/min) | 3,692 | 3,331 | +10.8% worse, $3.61 absolute |

Reading: on a pure random walk there is no signal to find, so any
forecast adds error — the model still loses ~6% there. On real
data (which has persistent drift / mean reversion) it wins. The
flat case's relative gap looks large but is $3.61 in absolute terms.

## I/O, determinism, timing

- Exact grader format: input `{"case", "markets": [{"prices": [...]}]}`,
  output `{"forecasts": [{"case": <id>, "price": <n>}]}` — verified
  byte-identical to `forecast()` called directly, deterministic
  across two runs, one `.out` per `.in` (multiple files OK).
- Plain-numbers `.in` (not a grader format): valid JSON `.out`,
  fallback = last price (a scanner bug that dropped the final
  number was found and fixed in `main.py`).
- Fewer than 300 prices: returns the last price.
- Timing: **0.17–0.25 s per case** (limit 8 s), 1.11 s for all
  6 practice cases. Stdlib only (`json`, `math`, `os`, `pathlib`);
  no network, no random, no clock.

## Input-format assumptions

The exact format was confirmed from the kit (`competition.yaml`,
`input/NOTICE.txt`, `starter/main.ts`): one JSON object per `.in`,
`markets[0].prices` = 4,320 one-minute closes, last = checkpoint
close; output must be the JSON `forecasts` object — a plain number
is an **invalid** case (scored as persistence, entry takes no
prize). An earlier draft of `main.py` wrote a plain number; that
was caught by reading the kit and fixed before this report.
`main.py` still falls back to a number-scan + last price if a case
file is not valid JSON, so a malformed case still yields a validly
shaped (persistence) output instead of a crash.

## Caveats

- Practice is explicitly "no guide to the final" (kit NOTICE):
  the final's 12 cases are cut from the 48 h after the deadline,
  a different, unseen window. Beating persistence on practice and
  on recent Binance data is encouraging, not a guarantee.
- The 5-case synthetic / real samples are small; per-case errors
  are dominated by a few large 4 h jumps (e.g. −3.5% in one real
  case) that no 72 h history can predict.
- Scores were computed locally on Python 3.12.3; the grader uses
  Python 3.13 — the code uses only stable stdlib features.
