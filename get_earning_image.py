import datetime as _dt
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class EarningsMetrics:
    symbol: str
    date: str
    eps_actual: Optional[float]
    eps_estimate: Optional[float]
    rev_actual: Optional[float]
    rev_estimate: Optional[float]


def _parse_date_yyyy_mm_dd(s: str) -> _dt.date:
    return _dt.date.fromisoformat(s)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pct_diff(actual: Optional[float], estimate: Optional[float]) -> Optional[float]:
    if actual is None or estimate is None:
        return None
    if estimate == 0:
        return None
    return (actual - estimate) / abs(estimate) * 100.0


def _format_money(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    abs_v = abs(v)
    if abs_v >= 1_000_000_000:
        return f"{v/1_000_000_000:.2f}B"
    if abs_v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if abs_v >= 1_000:
        return f"{v/1_000:.2f}K"
    return f"{v:.2f}"


def _format_eps(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.2f}"


def _format_pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}%"


def _find_font(preferred: list[str]) -> Optional[str]:
    for p in preferred:
        if pathlib.Path(p).exists():
            return p
    return None


def _finnhub_api_key() -> Optional[str]:
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not key or key.upper().startswith("DUMMY"):
        return None
    return key


def _client():
    try:
        import finnhub  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: finnhub-python (pip install finnhub-python)") from e

    key = _finnhub_api_key()
    if key is None:
        return None
    return finnhub.Client(api_key=key)


def _normalize_quarter(q: Any) -> Optional[int]:
    if q is None:
        return None
    if isinstance(q, int):
        return q if 1 <= q <= 4 else None
    s = str(q).strip().lower()
    if s.startswith("q"):
        s = s[1:]
    try:
        n = int(s)
    except ValueError:
        return None
    return n if 1 <= n <= 4 else None


def _pick_closest_by_period(rows: list[dict[str, Any]], target_date: _dt.date) -> Optional[dict[str, Any]]:
    best = None
    best_days = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        period = r.get("period") or r.get("date")
        if not isinstance(period, str):
            continue
        try:
            d = _parse_date_yyyy_mm_dd(period)
        except ValueError:
            continue
        days = abs((d - target_date).days)
        if best is None or best_days is None or days < best_days:
            best = r
            best_days = days
    return best


def _pick_best_yq(rows: list[dict[str, Any]], year: Optional[int], quarter: Optional[int]) -> Optional[dict[str, Any]]:
    if year is None or quarter is None:
        return None
    for r in rows:
        if not isinstance(r, dict):
            continue
        # Revenue estimate rows often have a "quarter" int and "year" int.
        try:
            ry = int(r.get("year"))
            rq = int(r.get("quarter"))
        except Exception:
            continue
        if ry == year and rq == quarter:
            return r
    return None


def _extract_revenue_from_financials_reported(data: dict[str, Any]) -> Optional[float]:
    # Keep this minimal: scan the income statement for a revenue-like line.
    report = data.get("report")
    if not isinstance(report, dict):
        return None
    ic = report.get("ic")
    if not isinstance(ic, list):
        return None
    for row in ic:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").lower()
        concept = str(row.get("concept") or "").lower()
        if "revenue" in label or "revenues" in label or concept.endswith(":revenues") or "sales" in label:
            v = _safe_float(row.get("value"))
            if v is not None:
                return v
    return None


def _fetch_earnings_metrics_from_finnhub(symbol: str, earnings_date: str, *, year: Optional[int], quarter: Optional[int]) -> Optional[EarningsMetrics]:
    client = _client()
    if client is None:
        return None

    target = _parse_date_yyyy_mm_dd(earnings_date)

    eps_actual = None
    eps_estimate = None
    try:
        rows = client.company_earnings(symbol, limit=20)  # type: ignore[no-untyped-call]
        if isinstance(rows, list) and rows:
            picked = _pick_closest_by_period(rows, target)
            if isinstance(picked, dict):
                eps_actual = _safe_float(picked.get("actual"))
                eps_estimate = _safe_float(picked.get("estimate"))
    except Exception:
        pass

    rev_estimate = None
    try:
        res = client.company_revenue_estimates(symbol, freq="quarterly")  # type: ignore[no-untyped-call]
        rows = res.get("data") if isinstance(res, dict) else None
        if isinstance(rows, list) and rows:
            picked = _pick_best_yq(rows, year, quarter) or _pick_closest_by_period(rows, target)
            if isinstance(picked, dict):
                rev_estimate = _safe_float(picked.get("estimate") or picked.get("avg") or picked.get("consensus"))
    except Exception:
        pass

    rev_actual = None
    try:
        res = client.financials_reported(symbol=symbol, freq="quarterly")  # type: ignore[no-untyped-call]
        rows = res.get("data") if isinstance(res, dict) else None
        if isinstance(rows, list) and rows:
            picked = None
            if year is not None and quarter is not None:
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    try:
                        iy = int(item.get("year"))
                        iq = int(item.get("quarter"))
                    except Exception:
                        continue
                    if iy == year and iq == quarter:
                        picked = item
                        break
            if picked is None:
                picked = rows[0] if isinstance(rows[0], dict) else None
            if isinstance(picked, dict):
                rev_actual = _extract_revenue_from_financials_reported(picked)
    except Exception:
        pass

    if eps_actual is None and eps_estimate is None and rev_actual is None and rev_estimate is None:
        return None

    return EarningsMetrics(
        symbol=symbol.upper(),
        date=earnings_date,
        eps_actual=eps_actual,
        eps_estimate=eps_estimate,
        rev_actual=rev_actual,
        rev_estimate=rev_estimate,
    )


def _render_image(
    *,
    company_name: str,
    symbol: str,
    date: str,
    eps_actual: Optional[float],
    eps_estimate: Optional[float],
    rev_actual: Optional[float],
    rev_estimate: Optional[float],
    out_path: str,
) -> str:
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1280, 720
    bg = (10, 15, 28)  # dark navy
    fg = (242, 244, 247)  # near-white
    muted = (156, 163, 175)  # gray
    green = (34, 197, 94)
    red = (239, 68, 68)
    card = (17, 24, 39)
    border = (31, 41, 55)

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    font_regular_path = _find_font(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
    )
    font_bold_path = _find_font(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        ]
    )
    font_title = ImageFont.truetype(font_bold_path, 54) if font_bold_path else ImageFont.load_default()
    font_sub = ImageFont.truetype(font_regular_path, 28) if font_regular_path else ImageFont.load_default()
    font_label = ImageFont.truetype(font_bold_path, 30) if font_bold_path else ImageFont.load_default()
    font_value = ImageFont.truetype(font_bold_path, 44) if font_bold_path else ImageFont.load_default()

    pad = 56
    header_y = 44
    title = f"{company_name} ({symbol})"
    draw.text((pad, header_y), title, fill=fg, font=font_title)
    draw.text((pad, header_y + 70), f"Earnings date: {date}", fill=muted, font=font_sub)

    # table header
    table_top = 180
    table_left = pad
    table_right = W - pad
    row_h = 190
    col0_w = 220
    col_w = (table_right - table_left - col0_w) // 3
    cols_x = [
        table_left,
        table_left + col0_w,
        table_left + col0_w + col_w,
        table_left + col0_w + 2 * col_w,
        table_left + col0_w + 3 * col_w,
    ]

    headers = ["", "Actual", "Estimate", "Diff"]
    for i, htxt in enumerate(headers):
        x = cols_x[i] + 16
        draw.text((x, table_top - 46), htxt, fill=muted, font=font_label)

    def draw_row(y: int, label: str, actual: Optional[float], estimate: Optional[float], is_money: bool) -> None:
        # card background
        draw.rounded_rectangle(
            [table_left, y, table_right, y + row_h],
            radius=22,
            fill=card,
            outline=border,
            width=2,
        )

        draw.text((table_left + 22, y + 22), label, fill=fg, font=font_label)

        pct = _pct_diff(actual, estimate)
        pct_color = muted if pct is None else (green if pct > 0 else red if pct < 0 else muted)

        fmt = _format_money if is_money else _format_eps
        values = [fmt(actual), fmt(estimate), _format_pct(pct)]
        colors = [fg, fg, pct_color]

        for j in range(3):
            x0 = cols_x[j + 1]
            cx = x0 + col_w // 2
            txt = values[j]
            tw, th = draw.textbbox((0, 0), txt, font=font_value)[2:]
            draw.text((cx - tw / 2, y + 80), txt, fill=colors[j], font=font_value)

    draw_row(table_top, "EPS", eps_actual, eps_estimate, False)
    draw_row(table_top + row_h + 36, "Revenue", rev_actual, rev_estimate, True)

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path


def create_earnings_summary_image(company: dict, *, summary_text: str | None = None, out_path: str | None = None) -> str:
    """
    Create a 1280×720 PNG summarizing the company's quarterly earnings vs estimates.

    Data source: Finnhub earnings calendar when FINNHUB_API_KEY is set.
    If data is unavailable, the image still renders with N/A values.
    """
    _ = summary_text  # reserved for future subtitle usage

    ticker = str(company.get("ticker") or "").strip().upper()
    name = str(company.get("name") or ticker or "Company").strip()
    earnings_call = company.get("earnings_call") or {}
    date = str(earnings_call.get("date") or "").strip()
    year = earnings_call.get("year")
    try:
        year_i = int(year) if year is not None else None
    except Exception:
        year_i = None
    quarter_i = _normalize_quarter(earnings_call.get("quarter"))
    if not ticker:
        raise ValueError("company.ticker is required")
    if not date:
        # fallback: today (still renders, and Finnhub query will just pick nearest)
        date = _dt.date.today().isoformat()

    metrics = _fetch_earnings_metrics_from_finnhub(ticker, date, year=year_i, quarter=quarter_i)
    if metrics is None:
        metrics = EarningsMetrics(
            symbol=ticker,
            date=date,
            eps_actual=None,
            eps_estimate=None,
            rev_actual=None,
            rev_estimate=None,
        )

    if out_path is None:
        fd, p = tempfile.mkstemp(suffix=f"_{ticker}.png")
        os.close(fd)
        out_path = p

    return _render_image(
        company_name=name,
        symbol=metrics.symbol,
        date=metrics.date,
        eps_actual=metrics.eps_actual,
        eps_estimate=metrics.eps_estimate,
        rev_actual=metrics.rev_actual,
        rev_estimate=metrics.rev_estimate,
        out_path=out_path,
    )


if __name__ == "__main__":
    # Quick manual test (renders N/A unless FINNHUB_API_KEY is set and endpoints are available).
    sample_company = {
        "name": "Apple",
        "ticker": "AAPL",
        "earnings_call": {"date": "2026-01-29", "year": 2026, "quarter": "q1"},
    }
    print(create_earnings_summary_image(sample_company))

