import argparse
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _http_get_text(url: str, *, user_agent: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json,text/html,*/*",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def _normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def _ticker_candidates(ticker: str) -> list[str]:
    """
    Candidates for earningscall.biz 'symbol' path segment.
    Keep it simple and deterministic.
    """
    t = _normalize_ticker(ticker).lower()
    out = [t]
    if "." in t:
        out.append(t.replace(".", "-"))
        out.append(t.replace(".", ""))
    # common special-case used by earningscall.biz
    if t == "googl":
        out.append("goog")
    return list(dict.fromkeys(out))


def _cik_to_10_digits(cik: int | str) -> str:
    s = str(cik).strip()
    if not s.isdigit():
        raise ValueError(f"CIK must be digits, got {cik!r}")
    return s.zfill(10)


@dataclass(frozen=True)
class EarningsCallMeta:
    exchange: str
    symbol: str
    year: int
    quarter: str  # q1..q4
    call_date: str | None  # YYYY-MM-DD


def _discover_latest_earnings_call(
    *,
    ticker: str,
    user_agent: str,
    exchanges: list[str] | None = None,
) -> EarningsCallMeta | None:
    exchanges = exchanges or ["nasdaq", "nyse", "amex"]

    def first_yq(listing_html: str) -> tuple[int, str] | None:
        m = re.search(r"/y/(\d{4})/q/(q[1-4])\b", listing_html)
        if not m:
            return None
        return int(m.group(1)), m.group(2)

    def first_mmddyyyy(page_html: str) -> str | None:
        m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", page_html)
        if not m:
            return None
        mm, dd, yy = map(int, m.groups())
        return date(yy, mm, dd).isoformat()

    for symbol in _ticker_candidates(ticker):
        for ex in exchanges:
            listing_url = f"https://earningscall.biz/e/{ex}/s/{symbol}"
            try:
                listing_html = _http_get_text(listing_url, user_agent=user_agent)
            except Exception:
                continue

            yq = first_yq(listing_html)
            if not yq:
                continue

            year, quarter = yq
            transcript_url = f"https://earningscall.biz/e/{ex}/s/{symbol}/y/{year}/q/{quarter}"
            try:
                transcript_html = _http_get_text(transcript_url, user_agent=user_agent)
            except Exception:
                continue

            return EarningsCallMeta(
                exchange=ex,
                symbol=symbol,
                year=year,
                quarter=quarter,
                call_date=first_mmddyyyy(transcript_html),
            )

    return None


def _load_sec_company_tickers(*, user_agent: str) -> list[dict[str, Any]]:
    """
    Loads the SEC mapping file once.
    Shape example:
      {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    """
    text = _http_get_text(SEC_TICKERS_URL, user_agent=user_agent)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Unexpected SEC tickers JSON shape")

    rows: list[dict[str, Any]] = []
    for _, rec in data.items():
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _sec_lookup_by_ticker(sec_rows: list[dict[str, Any]], ticker: str) -> dict[str, Any] | None:
    wanted = _normalize_ticker(ticker)

    def norm(s: str) -> str:
        s = _normalize_ticker(s)
        # try to be forgiving for dot vs dash
        return s.replace(".", "-")

    wanted2 = norm(wanted)

    for rec in sec_rows:
        tk = rec.get("ticker")
        if not tk:
            continue
        if norm(str(tk)) == wanted2:
            return rec
    return None


def validate_companies_json(data: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["root must be an object"]
    companies = data.get("companies")
    if not isinstance(companies, list):
        return ["root.companies must be a list"]

    seen: set[str] = set()
    for i, c in enumerate(companies):
        if not isinstance(c, dict):
            errors.append(f"companies[{i}] must be an object")
            continue

        ticker = c.get("ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            errors.append(f"companies[{i}].ticker is required")
        else:
            tk = _normalize_ticker(ticker)
            if tk in seen:
                errors.append(f"duplicate ticker: {tk}")
            seen.add(tk)

        cik = c.get("cik")
        if cik is not None:
            if not isinstance(cik, str) or not cik.isdigit() or len(cik) != 10:
                errors.append(f"companies[{i}].cik must be 10-digit string (got {cik!r})")

        exchange = c.get("exchange")
        if exchange is not None and (not isinstance(exchange, str) or not exchange.strip()):
            errors.append(f"companies[{i}].exchange must be a non-empty string if present")

        ec = c.get("earnings_call")
        if ec is not None:
            if not isinstance(ec, dict):
                errors.append(f"companies[{i}].earnings_call must be object or null")
            else:
                for k in ("symbol", "year", "quarter"):
                    if k not in ec:
                        errors.append(f"companies[{i}].earnings_call.{k} is required when earnings_call is set")
                q = ec.get("quarter")
                if isinstance(q, str) and not re.fullmatch(r"q[1-4]", q):
                    errors.append(f"companies[{i}].earnings_call.quarter must be q1..q4 (got {q!r})")
                d = ec.get("date")
                if d is not None:
                    if not isinstance(d, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
                        errors.append(f"companies[{i}].earnings_call.date must be YYYY-MM-DD or null (got {d!r})")

    return errors


def _load_companies_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_companies_file(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _find_company_index(companies: list[dict], ticker: str) -> int | None:
    wanted = _normalize_ticker(ticker)
    for i, c in enumerate(companies):
        if _normalize_ticker(c.get("ticker", "")) == wanted:
            return i
    return None


def add_or_update_by_tickers(
    *,
    companies_path: str,
    tickers: list[str],
    user_agent: str,
    update_existing: bool,
    dry_run: bool,
) -> int:
    data = _load_companies_file(companies_path)
    errors = validate_companies_json(data)
    if errors:
        raise SystemExit("companies.json validation failed:\n- " + "\n- ".join(errors))

    companies: list[dict] = data["companies"]
    sec_rows = _load_sec_company_tickers(user_agent=user_agent)

    changed = False
    for t in tickers:
        tk = _normalize_ticker(t)
        idx = _find_company_index(companies, tk)
        if idx is None:
            companies.append({"ticker": tk})
            idx = len(companies) - 1
            changed = True

        company = companies[idx]

        # SEC enrichment (name + cik)
        sec = _sec_lookup_by_ticker(sec_rows, tk)
        if sec:
            if update_existing or not company.get("name"):
                if sec.get("title"):
                    company["name"] = str(sec["title"])
                    changed = True
            if update_existing or not company.get("cik"):
                if sec.get("cik_str") is not None:
                    company["cik"] = _cik_to_10_digits(sec["cik_str"])
                    changed = True

        # EarningsCall enrichment (exchange + earnings_call meta)
        ec = _discover_latest_earnings_call(ticker=tk, user_agent=user_agent)
        if ec:
            if update_existing or not company.get("exchange"):
                company["exchange"] = ec.exchange
                changed = True
            if update_existing or not company.get("earnings_call"):
                company["earnings_call"] = {
                    "symbol": ec.symbol,
                    "year": ec.year,
                    "quarter": ec.quarter,
                    "date": ec.call_date,
                }
                changed = True

    # Sort by ticker for readability
    companies.sort(key=lambda c: _normalize_ticker(c.get("ticker", "")))
    data["companies"] = companies

    # Validate after modifications
    errors2 = validate_companies_json(data)
    if errors2:
        raise SystemExit("post-update validation failed:\n- " + "\n- ".join(errors2))

    if changed and not dry_run:
        _save_companies_file(companies_path, data)

    return 0


def _parse_tickers(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="companies.json", help="Path to companies.json")
    ap.add_argument(
        "--ticker",
        action="append",
        default=[],
        help="Ticker(s) to add/update. Can be repeated or comma-separated.",
    )
    ap.add_argument("--validate-only", action="store_true", help="Only validate companies.json and exit")
    ap.add_argument("--update-existing", action="store_true", help="Overwrite existing name/cik/exchange/earnings_call")
    ap.add_argument("--dry-run", action="store_true", help="Do not write changes")
    ap.add_argument(
        "--user-agent",
        default="finance-podcast-bot companies_updater/1.0 (contact: you@example.com)",
        help="User-Agent for SEC and earningscall.biz",
    )
    args = ap.parse_args()

    data = _load_companies_file(args.file)
    errs = validate_companies_json(data)
    if errs:
        print("INVALID companies.json", file=sys.stderr)
        for e in errs:
            print(f"- {e}", file=sys.stderr)
        return 2
    if args.validate_only:
        print("companies.json valid")
        return 0

    tickers = _parse_tickers(args.ticker)
    if not tickers:
        ap.error("provide --ticker TICKER (at least one), or use --validate-only")

    return add_or_update_by_tickers(
        companies_path=args.file,
        tickers=tickers,
        user_agent=args.user_agent,
        update_existing=args.update_existing,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())

