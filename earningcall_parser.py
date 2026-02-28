import argparse
import json
import html
import os
import re
import sys
import urllib.request
from html.parser import HTMLParser


class _HtmlToText(HTMLParser):
    _BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "main",
        "nav",
        "aside",
        "br",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ul",
        "ol",
        "pre",
        "blockquote",
        "tr",
        "td",
        "th",
        "table",
    }

    _SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if data:
            self._chunks.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._skip_depth:
            return
        self._chunks.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._skip_depth:
            return
        self._chunks.append(f"&#{name};")

    def text(self) -> str:
        return "".join(self._chunks)


def _normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def _http_get_json(url: str, *, user_agent: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent, "Accept": "application/json,*/*"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
    return json.loads(raw.decode(charset, errors="replace"))


def _sec_cik_for_ticker(sec_company_tickers: dict, ticker: str) -> int | None:
    wanted = _normalize_ticker(ticker).replace(".", "-")
    for rec in sec_company_tickers.values():
        if not isinstance(rec, dict):
            continue
        tk = rec.get("ticker")
        if not tk:
            continue
        if _normalize_ticker(str(tk)).replace(".", "-") == wanted:
            cik = rec.get("cik_str")
            try:
                return int(cik)
            except Exception:
                return None
    return None


def _normalize_quarter(quarter: int | str) -> str:
    q = str(quarter).strip().lower()
    if q.startswith("q"):
        q = q[1:]
    if q not in {"1", "2", "3", "4"}:
        raise ValueError(f"quarter must be 1-4 or q1-q4, got: {quarter!r}")
    return f"q{q}"


def _build_url(exchange: str, ticker: str, year: int, quarter: int | str) -> str:
    ex = str(exchange).strip().lower()
    if not ex:
        raise ValueError("exchange is required (e.g. 'nasdaq')")
    tk = str(ticker).strip().lower()
    if not tk:
        raise ValueError("ticker is required (e.g. 'msft')")
    y = int(year)
    q = _normalize_quarter(quarter)
    return f"https://earningscall.biz/e/{ex}/s/{tk}/y/{y}/q/{q}"


def _fetch_html(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) earningscall-parser/1.0",
            "Accept": "text/html,*/*",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def _clean_transcript_text(text: str) -> str:
    s = html.unescape(text)
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize whitespace but keep paragraph breaks.
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    lines = [ln.strip() for ln in s.split("\n")]

    # Drop empty lines and obvious chrome lines.
    chrome_exact = {
        "search",
        "calendar",
        "chatai",
        "pricing",
        "resources",
        "about us",
        "top employers",
        "login",
        "download app",
        "download apps",
        "designed by",
        "company",
        "quick link",
        "resource",
        "download",
        "share",
        "disclaimer",
    }

    def is_noise(ln: str) -> bool:
        if not ln:
            return True
        low = ln.lower()
        if low.startswith("earningscall ·"):
            return True
        if low in chrome_exact:
            return True
        if low in {"-", "–", "—", "1.0x"}:
            return True
        if re.fullmatch(r"©.*", ln):
            return True
        return False

    # Keep everything, but trim top chrome.
    cleaned: list[str] = []
    started = False
    for ln in lines:
        if not started:
            if is_noise(ln):
                continue
            # Most transcripts start at "Operator". If not, we still start at first non-noise.
            started = True
        cleaned.append(ln)

    # Prefer starting at "Operator" if present.
    for i, ln in enumerate(cleaned):
        if ln.lower() == "operator":
            cleaned = cleaned[i:]
            break

    # Stop at Disclaimer if present (keep transcript above it).
    for i, ln in enumerate(cleaned):
        if ln.lower() == "disclaimer":
            cleaned = cleaned[:i]
            break

    # Final pass: remove extra blank runs.
    out_lines: list[str] = []
    last_blank = False
    for ln in cleaned:
        blank = not ln
        if blank and last_blank:
            continue
        out_lines.append(ln)
        last_blank = blank

    return "\n".join(out_lines).strip() + "\n"


def parse_earningscall_transcript(exchange: str, ticker: str, year: int, quarter: int | str) -> str:
    """
    Fetch and parse an earnings call transcript from earningscall.biz into plain text.

    Returns a single cleaned text string including Q&A (if present).
    """
    url = _build_url(exchange=exchange, ticker=ticker, year=year, quarter=quarter)
    html_doc = _fetch_html(url)
    parser = _HtmlToText()
    parser.feed(html_doc)
    parser.close()
    return _clean_transcript_text(parser.text())


def parse_company_earningscall_transcript(company: dict) -> str:
    """
    Parse transcript using a company entry from companies.json.

    Expected shape:
      - company['earnings_call'] = {symbol, year, quarter, date} OR null
      - company['exchange'] (fallback only; actual earnings call exchange should match the site URL)
    """
    ec = company.get("earnings_call")
    if not ec:
        raise ValueError(f"No earnings_call metadata for {company.get('ticker')}")

    exchange = (ec.get("exchange") or company.get("exchange") or "").strip()
    symbol = (ec.get("symbol") or company.get("ticker") or "").strip()
    year = ec["year"]
    quarter = ec["quarter"]
    if not exchange:
        raise ValueError(f"Missing exchange for {company.get('ticker')}")
    if not symbol:
        raise ValueError(f"Missing earnings_call symbol for {company.get('ticker')}")
    return parse_earningscall_transcript(exchange=exchange, ticker=symbol, year=year, quarter=quarter)


def get_latest_10q_info(ticker: str, sec_identity: str) -> dict:
    """
    Simple SEC 10-Q metadata lookup (no API key, stdlib-only).

    Fetches:
    - SEC ticker->CIK mapping
    - SEC company submissions JSON

    Returns a dict with keys:
      - ticker
      - cik (10-digit string)
      - company_name
      - period_of_report (reportDate)
      - filing_date
      - accession_number
      - primary_document
      - filing_index_url
      - primary_document_url
    """
    if not sec_identity:
        raise ValueError("sec_identity is required (SEC requires a descriptive User-Agent/contact)")

    user_agent = f"finance-podcast-bot ({sec_identity})"

    # 1) Ticker -> CIK (SEC-maintained mapping)
    tickers_json = _http_get_json("https://www.sec.gov/files/company_tickers.json", user_agent=user_agent)
    cik_int = _sec_cik_for_ticker(tickers_json, ticker)
    if cik_int is None:
        raise ValueError(f"Ticker not found in SEC mapping: {ticker}")
    cik10 = str(int(cik_int)).zfill(10)

    # 2) Company submissions -> latest 10-Q
    submissions = _http_get_json(f"https://data.sec.gov/submissions/CIK{cik10}.json", user_agent=user_agent)
    company_name = submissions.get("name")
    recent = (submissions.get("filings") or {}).get("recent") or {}

    forms = recent.get("form") or []
    accession = recent.get("accessionNumber") or []
    filing_date = recent.get("filingDate") or []
    report_date = recent.get("reportDate") or []
    primary_doc = recent.get("primaryDocument") or []

    if not (isinstance(forms, list) and isinstance(accession, list) and isinstance(filing_date, list)):
        raise ValueError("Unexpected SEC submissions JSON shape (recent arrays missing)")

    idx = None
    for i, f in enumerate(forms):
        if f == "10-Q":
            idx = i
            break
    if idx is None:
        raise ValueError(f"No 10-Q found in recent filings for {ticker}")

    acc = accession[idx]
    acc_nodash = str(acc).replace("-", "")
    cik_noleading = str(int(cik10))
    base = f"https://www.sec.gov/Archives/edgar/data/{cik_noleading}/{acc_nodash}/"

    info = {
        "ticker": _normalize_ticker(ticker),
        "cik": cik10,
        "company_name": company_name,
        "period_of_report": report_date[idx] if idx < len(report_date) else None,
        "filing_date": filing_date[idx],
        "accession_number": acc,
        "primary_document": primary_doc[idx] if idx < len(primary_doc) else None,
        "filing_index_url": base + "index.json",
        "primary_document_url": base + (primary_doc[idx] if idx < len(primary_doc) else ""),
    }
    return info


def compare_latest_10q_across_companies(
    tickers: list[str],
    sec_identity: str,
) -> list[dict]:
    """
    Compare latest 10-Q metadata across companies.
    """
    out: list[dict] = []
    for t in tickers:
        try:
            info = get_latest_10q_info(t, sec_identity)
            out.append(
                {
                    "ticker": t,
                    "period_of_report": info.get("period_of_report"),
                    "filing_date": info.get("filing_date"),
                    "cik": info.get("cik"),
                    "accession_number": info.get("accession_number"),
                    "filing_index_url": info.get("filing_index_url"),
                }
            )
        except Exception as e:
            out.append({"ticker": t, "error": str(e)})
    return out


def _ticker_from_companies_json(path: str, company_index: int) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    companies = data["companies"]
    if not isinstance(companies, list) or not companies:
        raise ValueError(f"{path} has no companies list")
    if company_index < 0 or company_index >= len(companies):
        raise IndexError(f"company_index out of range: {company_index} (0..{len(companies) - 1})")
    ticker = str(companies[company_index].get("ticker", "")).strip()
    if not ticker:
        raise ValueError(f"companies[{company_index}].ticker is missing/empty in {path}")
    return ticker


def _company_from_companies_json(path: str, company_index: int) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    companies = data["companies"]
    if not isinstance(companies, list) or not companies:
        raise ValueError(f"{path} has no companies list")
    if company_index < 0 or company_index >= len(companies):
        raise IndexError(f"company_index out of range: {company_index} (0..{len(companies) - 1})")
    return companies[company_index]


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", help="e.g. nasdaq / nyse (optional if --company-index is provided)")
    ap.add_argument("--ticker", help="e.g. MSFT (optional if --company-index is provided)")
    ap.add_argument("--year", type=int, help="e.g. 2026 (optional if --company-index is provided)")
    ap.add_argument("--quarter", help="1-4 or q1-q4 (optional if --company-index is provided)")
    ap.add_argument("--companies-json", default="companies.json", help="Path to companies.json (default: companies.json)")
    ap.add_argument(
        "--all-companies",
        action="store_true",
        help="Parse all companies in companies.json that have earnings_call metadata.",
    )
    tenq_group = ap.add_mutually_exclusive_group()
    tenq_group.add_argument(
        "--with-10q",
        dest="with_10q",
        action="store_true",
        help="(Default) Print latest 10-Q metadata from SEC for each company.",
    )
    tenq_group.add_argument(
        "--no-10q",
        dest="with_10q",
        action="store_false",
        help="Disable printing 10-Q metadata.",
    )
    ap.set_defaults(with_10q=True)
    ap.add_argument(
        "--sec-identity",
        default="",
        help="SEC identity string/email for edgartools (or set env SEC_IDENTITY).",
    )
    ap.add_argument(
        "--company-index",
        type=int,
        default=None,
        help="Use ticker from companies.json at this index (0-based). Overrides --ticker.",
    )
    args = ap.parse_args()

    if not args.sec_identity:
        args.sec_identity = os.environ.get("SEC_IDENTITY", "my@email.com")

    if args.all_companies:
        with open(args.companies_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        companies = data.get("companies") or []
        if not isinstance(companies, list) or not companies:
            ap.error("companies.json has no companies list")

        interactive = sys.stdin.isatty()
        for company in companies:
            ticker = (company.get("ticker") or "").strip()
            if not company.get("earnings_call"):
                continue

            if args.with_10q:
                try:
                    info = get_latest_10q_info(ticker, sec_identity=args.sec_identity)
                    print(
                        f"[10-Q] {ticker} "
                        f"period={info.get('period_of_report')} filed={info.get('filing_date')} "
                        f"acc={info.get('accession_number')} cik={info.get('cik')}"
                    )
                except Exception as e:
                    print(f"[10-Q ERROR] {ticker}: {e}", file=sys.stderr)

            try:
                text = parse_company_earningscall_transcript(company)
            except Exception as e:
                print(f"[ERROR] {ticker}: {e}", file=sys.stderr)
                continue

            header = f"===== {ticker} ====="
            print(header)
            print(text, end="")
            print()

            if interactive:
                try:
                    answer = input("Press Enter to continue (or 'q' to quit): ").strip().lower()
                except EOFError:
                    answer = "q"
                if answer in {"q", "quit", "exit"}:
                    break
        return 0

    if args.company_index is not None:
        company = _company_from_companies_json(args.companies_json, args.company_index)
        exchange = args.exchange or company.get("exchange")
        year = args.year or (company.get("earnings_call") or {}).get("year")
        quarter = args.quarter or (company.get("earnings_call") or {}).get("quarter")
        ticker = args.ticker or (company.get("earnings_call") or {}).get("symbol") or company.get("ticker")
    else:
        exchange = args.exchange
        year = args.year
        quarter = args.quarter
        ticker = args.ticker

    if not exchange or not year or not quarter or not ticker:
        ap.error("provide --exchange/--ticker/--year/--quarter, or use --company-index with earnings_call metadata in companies.json")

    text = parse_earningscall_transcript(
        exchange=exchange,
        ticker=ticker,
        year=year,
        quarter=quarter,
    )
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

