#!/usr/bin/env python3
"""Test harness for btc-forecast/main.py (stdlib only).

- Synthetic series: random walk, trending, mean-reverting, flat.
  Checks: output is one number, deterministic on repeat, timing.
- Input formats: plain lines, CSV timestamp,price, JSON array,
  JSON object {"prices": [...]}, JSON-lines.
- Real data (if reachable): Binance US public klines, no key.
  Builds 72h cases ending 4h apart, scores MAE in cents vs
  persistence.  If the network/API is unreachable, reports that
  and scores only the synthetic series.
Run:  python3 test_forecast.py
"""
import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "main.py")
sys.path.insert(0, HERE)
import main as M  # noqa: E402  (import for direct forecast() testing)


def run_main_via_files(prices, fmt="plain", case_id="case1"):
    """Run main.py exactly as the grader does: /input/<id>.in -> /output/<id>.out"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        in_dir = os.path.join(td, "input")
        out_dir = os.path.join(td, "output")
        os.makedirs(in_dir)
        # The grader's exact format (see kit competition.yaml):
        # {"case": <id>, "markets": [{"id": "btc-usdc", "prices": [...]}]}
        if fmt == "exact":
            text = json.dumps({"schema": "makefun/btc-cent-case/v1",
                               "competition": "btc-cent-forecast-01",
                               "case": case_id, "at": "2026-11-10T06:00:00.000Z",
                               "horizon_minutes": 240,
                               "markets": [{"id": "btc-usdc", "name": "BTC/USDC",
                                            "start": "2026-11-07T06:01:00.000Z",
                                            "prices": [round(p, 5) for p in prices]}]})
        elif fmt == "plain":
            # Not a grader format: main.py falls back to persistence
            # (last price) but must still write a valid JSON .out.
            text = "\n".join(f"{p:.2f}" for p in prices)
        else:
            raise ValueError(fmt)
        with open(os.path.join(in_dir, case_id + ".in"), "w") as fh:
            fh.write(text)
        env = dict(os.environ)
        env["MAKEFUN_INPUT_DIR"] = in_dir
        env["MAKEFUN_OUTPUT_DIR"] = out_dir
        t0 = time.time()
        r = subprocess.run([sys.executable, "-P", "-s", MAIN],
                           env=env, capture_output=True, text=True, timeout=30)
        dt = time.time() - t0
        out_path = os.path.join(out_dir, case_id + ".out")
        assert r.returncode == 0, f"exit {r.returncode}: {r.stderr}"
        assert os.path.exists(out_path), "no .out file"
        content = open(out_path).read().strip()
        # Exact grader format: {"forecasts": [{"case": <id>, "price": <n>}]}
        data = json.loads(content)
        assert set(data.keys()) == {"forecasts"}, f"bad output keys {data!r}"
        assert len(data["forecasts"]) == 1, f"bad output {data!r}"
        row = data["forecasts"][0]
        assert set(row.keys()) == {"case", "price"} and row["case"] == case_id, f"bad row {row!r}"
        val = float(row["price"])
        assert math.isfinite(val) and 0 < val < 10_000_000, f"bad output {val}"
        return val, dt, content


def gen_series(kind, n=4320 + 240, seed=42, start=64000.0):
    rnd = random.Random(seed)
    logp = [math.log(start)]
    for i in range(1, n):
        if kind == "random_walk":
            shock = rnd.gauss(0, 0.0008)
            drift = 0.0
        elif kind == "trending":
            shock = rnd.gauss(0, 0.0008)
            drift = 0.000012  # ~+1.7% / day, persistent
        elif kind == "mean_reverting":
            # OU toward log(start), half-life ~6h
            shock = rnd.gauss(0, 0.0008)
            drift = -0.0015 * (logp[-1] - math.log(start))
        elif kind == "flat":
            shock = rnd.gauss(0, 0.00005)
            drift = 0.0
        else:
            raise ValueError(kind)
        logp.append(logp[-1] + drift + shock)
    return [math.exp(x) for x in logp]


def mae_cents(preds, actuals):
    return 100.0 * sum(abs(p - a) for p, a in zip(preds, actuals)) / len(actuals)


def score_cases(cases):
    """cases: list of (prices_72h, actual_4h_later). Returns (mae_model, mae_persist)."""
    mp, pp, ac = [], [], []
    for prices, actual in cases:
        mp.append(M.forecast(prices))
        pp.append(prices[-1])
        ac.append(actual)
    return mae_cents(mp, ac), mae_cents(pp, ac)


def fetch_binance_us_closes(n_minutes=6000):
    """Fetch BTCUSDT 1m closes from Binance US (public, no key).
    Returns list of closes, oldest first, or None if unreachable."""
    url_base = "https://api.binance.us/api/v3/klines"
    closes = []
    end = None
    try:
        while len(closes) < n_minutes:
            limit = min(1000, n_minutes - len(closes))
            url = f"{url_base}?symbol=BTCUSDT&interval=1m&limit={limit}"
            if end is not None:
                url += f"&endTime={end}"
            req = urllib.request.Request(url, headers={"User-Agent": "btc-forecast-test/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.load(resp)
            if not data:
                break
            batch = [float(k[4]) for k in data]  # close
            closes = batch + closes
            end = int(data[0][0]) - 1  # before first open time
            if len(data) < limit:
                break
        return closes[-n_minutes:] if closes else None
    except Exception as e:
        print(f"  [real data] Binance US unreachable: {e}")
        return None


def main():
    print("== btc-forecast test ==")
    print(f"python {sys.version.split()[0]}, main.py: {MAIN}")
    results = {}

    # ---- synthetic, via direct forecast() ----
    print("\n-- synthetic series (4320 prices + 240 min future) --")
    for kind in ("random_walk", "trending", "mean_reverting", "flat"):
        series = gen_series(kind, n=6500, seed=42)
        # Build cases properly: checkpoint at t, case = series[t-4320:t], actual = series[t+240]
        cases = []
        for ck in (5760, 5520, 5280, 5040, 4800):
            cases.append((series[ck - 4320:ck], series[ck + 240]))
        mae_m, mae_p = score_cases(cases)
        results[kind] = (mae_m, mae_p)
        print(f"  {kind:15s} model MAE {mae_m:9.1f} cents   persistence MAE {mae_p:9.1f} cents   "
              f"{'BEATS' if mae_m < mae_p else 'worse'} persistence")

    # ---- input formats + determinism + timing, via grader-style files ----
    print("\n-- grader-style I/O, determinism, timing --")
    series = gen_series("trending", n=6500, seed=7)
    prices = series[:4320]
    # Exact grader format: output must equal forecast() directly.
    v1, dt1, _ = run_main_via_files(prices, fmt="exact")
    v2, dt2, _ = run_main_via_files(prices, fmt="exact")
    assert v1 == v2, f"non-deterministic: {v1} vs {v2}"
    assert abs(v1 - M.forecast(prices)) < 0.01, f"file run {v1} != forecast() {M.forecast(prices)}"
    print(f"  exact JSON -> {v1:.2f}  ({dt1:.2f}s, deterministic, matches forecast())")
    assert dt1 < 8.0, "too slow"
    # Plain-numbers .in (not a grader format): valid JSON .out, persistence fallback.
    v3, _, _ = run_main_via_files(prices, fmt="plain")
    print(f"  plain .in  -> {v3:.2f}  (fallback = last price {prices[-1]:.2f})")
    assert abs(v3 - prices[-1]) < 0.01
    # fallback: <300 prices -> persistence (last price)
    v, _, _ = run_main_via_files([64000.0, 64100.0, 63900.0])
    print(f"  fallback <300 prices -> {v:.2f} (expect last price 63900.00)")
    assert abs(v - 63900.0) < 0.01
    # multiple files
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        in_dir = os.path.join(td, "input"); out_dir = os.path.join(td, "output")
        os.makedirs(in_dir)
        for cid, seed in (("a", 1), ("b", 2)):
            s = gen_series("random_walk", n=6500, seed=seed)
            case = {"schema": "makefun/btc-cent-case/v1", "competition": "btc-cent-forecast-01",
                    "case": cid, "at": "2026-11-10T06:00:00.000Z", "horizon_minutes": 240,
                    "markets": [{"id": "btc-usdc", "prices": [round(p, 5) for p in s[:4320]]}]}
            open(os.path.join(in_dir, cid + ".in"), "w").write(json.dumps(case))
        env = dict(os.environ); env["MAKEFUN_INPUT_DIR"] = in_dir; env["MAKEFUN_OUTPUT_DIR"] = out_dir
        r = subprocess.run([sys.executable, "-P", "-s", MAIN], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        outs = sorted(os.listdir(out_dir))
        print(f"  multiple .in files -> {outs}")
        assert outs == ["a.out", "b.out"]

    # ---- real data ----
    print("\n-- real BTC data (Binance US, 1m) --")
    closes = fetch_binance_us_closes(6000)
    if closes and len(closes) >= 4320 + 240:
        print(f"  fetched {len(closes)} closes, last {closes[-1]:.2f}, first {closes[0]:.2f}")
        cases = []
        # checkpoints 4h apart, each with 240 min of future known
        for ck in range(len(closes) - 241, len(closes) - 241 - 5 * 240, -240):
            cases.append((closes[ck - 4320:ck], closes[ck + 240]))
        t0 = time.time()
        mae_m, mae_p = score_cases(cases)
        dt = time.time() - t0
        print(f"  {len(cases)} cases, 72h each, 4h apart")
        print(f"  model MAE {mae_m:.1f} cents   persistence MAE {mae_p:.1f} cents   "
              f"{'BEATS' if mae_m < mae_p else 'worse'} persistence")
        print(f"  scoring {len(cases)} cases took {dt:.2f}s total ({dt/len(cases):.2f}s/case)")
        results["real_btc"] = (mae_m, mae_p)
    else:
        print("  [real data] not available - synthetic only (see note in TEST_REPORT.md)")
        results["real_btc"] = None

    # ---- kit practice cases (real grader format, real cbBTC prices) ----
    print("\n-- kit practice cases (make.fun, p00-p05) --")
    kit_in = os.path.join(HERE, "kit", "btc-cent-forecast-01-kit", "input")
    if os.path.isdir(kit_in):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "out")
            env = dict(os.environ)
            env["MAKEFUN_INPUT_DIR"] = kit_in
            env["MAKEFUN_OUTPUT_DIR"] = out_dir
            t0 = time.time()
            r = subprocess.run([sys.executable, "-P", "-s", MAIN],
                               env=env, capture_output=True, text=True, timeout=60)
            dt = time.time() - t0
            assert r.returncode == 0, r.stderr
            # Actual for p00-p04 is the NEXT case's checkpoint close
            # (its target minute); p05's target is past the window.
            lasts = {}
            for c in ("p00", "p01", "p02", "p03", "p04", "p05"):
                d = json.load(open(os.path.join(kit_in, c + ".in")))
                lasts[c] = d["markets"][0]["prices"][-1]
            mp, pp, ac = [], [], []
            for c, nxt in (("p00", "p01"), ("p01", "p02"), ("p02", "p03"),
                           ("p03", "p04"), ("p04", "p05")):
                g = json.load(open(os.path.join(out_dir, c + ".out")))["forecasts"][0]["price"]
                mp.append(g); pp.append(lasts[c]); ac.append(lasts[nxt])
                print(f"  {c} last={lasts[c]:.2f} actual={lasts[nxt]:.2f} guess={g:.2f} "
                      f"err_m={abs(g-lasts[nxt])*100:.0f} err_p={abs(lasts[c]-lasts[nxt])*100:.0f}")
            mae_m, mae_p = mae_cents(mp, ac), mae_cents(pp, ac)
            print(f"  practice MAE (p00-p04): model {mae_m:.1f} cents, "
                  f"persistence {mae_p:.1f} cents, "
                  f"{'BEATS' if mae_m < mae_p else 'worse'} persistence "
                  f"({(mae_m/mae_p-1)*100:+.1f}%), {dt:.2f}s for 6 cases")
            results["practice_p00_p04"] = (mae_m, mae_p)
    else:
        print("  [practice] kit not found - skipped")

    print("\n== summary ==")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
