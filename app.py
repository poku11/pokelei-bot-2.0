# app.py
"""
Streamlit Vinted Scanner - fichier unique
- Collector asynchrone (httpx + lxml)
- Scoring engine (profit / velocity / risk / score)
- Interface Streamlit simple et robuste
Usage:
  pip install -r requirements.txt
  streamlit run app.py
Notes:
 - Adapte les sélecteurs XPath si Vinted change sa structure HTML.
 - Respecte les Terms of Service de Vinted (rate limits, pas d'automatisation d'achat).
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import math, re, asyncio, time
import httpx
from lxml import html as lxml_html

import streamlit as st
import pandas as pd

# -----------------------
# Configuration
# -----------------------
USER_AGENT = "Mozilla/5.0 (compatible; VintedBot/1.0; +https://example.com/bot)"
DEFAULT_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"}

DEFAULT_CONCURRENCY = 6
DEFAULT_TIMEOUT = 15.0

VINTED_FEES_PCT = 0.10
AVG_SHIPPING_COST = 5.0
DEFAULT_REFURB_COST = 3.0

MIN_PROFIT_DESIRABLE = 5.0
MAX_PROFIT_CONSIDERED = 200.0

WEIGHT_PROFIT = 0.40
WEIGHT_VELOCITY = 0.55
WEIGHT_RISK = 0.15

# -----------------------
# Utilitaires & scoring
# -----------------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def median(nums: List[float]) -> Optional[float]:
    if not nums:
        return None
    s = sorted(nums)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float((s[n//2 - 1] + s[n//2]) / 2.0)

def estimate_market_price(historical_prices: List[float]) -> Optional[float]:
    return median(historical_prices)

def compute_net_profit(asking_price: float,
                       market_price_est: Optional[float],
                       fees_pct: float = VINTED_FEES_PCT,
                       shipping: float = AVG_SHIPPING_COST,
                       refurb: float = DEFAULT_REFURB_COST) -> Optional[float]:
    if market_price_est is None:
        return None
    gross = market_price_est - asking_price
    fees = market_price_est * fees_pct
    net = gross - fees - shipping - refurb
    return round(net, 2)

def velocity_score(asking_price: float,
                   market_price_est: Optional[float],
                   posted_at: Optional[str] = None,
                   likes: int = 0,
                   views: int = 0,
                   brand_popularity: float = 0.5) -> float:
    if market_price_est is None:
        return 0.0
    price_ratio = asking_price / market_price_est if market_price_est > 0 else 1.0
    price_factor = clamp(1.5 - price_ratio, 0.0, 1.5) / 1.5

    recency_factor = 0.0
    if posted_at:
        try:
            dt = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
            recency_factor = clamp(1 - (delta_hours / (24 * 7)), 0.0, 1.0)
        except Exception:
            recency_factor = 0.0

    social = math.log(1 + likes + views / 20.0) / 5.0
    social = clamp(social, 0.0, 1.0)
    brand = clamp(brand_popularity, 0.0, 1.0)

    score = 0.5 * price_factor + 0.3 * recency_factor + 0.15 * social + 0.05 * brand
    return clamp(score, 0.0, 1.0)

def risk_penalty(photo_quality: float = 1.0,
                 ambiguous_brand: bool = False,
                 suspect_low_price: bool = False) -> float:
    r = 0.0
    if photo_quality < 0.5:
        r += 0.25
    if ambiguous_brand:
        r += 0.35
    if suspect_low_price:
        r += 0.25
    return clamp(r, 0.0, 1.0)

def final_score(net_profit: Optional[float],
                velocity: float,
                risk: float,
                min_profit: float = MIN_PROFIT_DESIRABLE,
                max_profit: float = MAX_PROFIT_CONSIDERED) -> Dict[str, Any]:
    if net_profit is None:
        profit_norm = 0.0
    else:
        profit_norm = (net_profit - min_profit) / (max_profit - min_profit)
        profit_norm = clamp(profit_norm, 0.0, 1.0)

    raw = WEIGHT_PROFIT * profit_norm + WEIGHT_VELOCITY * velocity - WEIGHT_RISK * risk
    raw = clamp(raw, 0.0, 1.0)
    score_pct = round(raw * 100, 2)

    return {
        "score": score_pct,
        "profit_normalized": round(profit_norm, 4),
        "velocity": round(velocity, 4),
        "risk": round(risk, 4),
        "net_profit": None if net_profit is None else round(net_profit, 2)
    }

def score_listing(listing: Dict[str, Any]) -> Dict[str, Any]:
    asking = float(listing.get("asking_price", 0.0))
    market_est = listing.get("market_price_est")
    if market_est is None:
        market_est = estimate_market_price(listing.get("historical_prices", []))
    netp = compute_net_profit(asking, market_est)
    vel = velocity_score(asking, market_est,
                         posted_at=listing.get("posted_at"),
                         likes=int(listing.get("likes", 0)),
                         views=int(listing.get("views", 0)),
                         brand_popularity=float(listing.get("brand_popularity", 0.5)))
    risk = risk_penalty(photo_quality=float(listing.get("photo_quality", 1.0)),
                        ambiguous_brand=bool(listing.get("ambiguous_brand", False)),
                        suspect_low_price=bool(listing.get("suspect_low_price", False)))

    final = final_score(netp, vel, risk)
    return {
        "asking_price": asking,
        "market_price_est": market_est,
        "net_profit": final["net_profit"],
        "score": final["score"],
        "components": {
            "profit_normalized": final["profit_normalized"],
            "velocity": final["velocity"],
            "risk": final["risk"]
        },
        "raw": {
            "velocity_raw": vel,
            "risk_raw": risk,
            "net_profit_raw": netp
        }
    }

# -----------------------
# Collector Async
# -----------------------
class CollectorAsync:
    def __init__(self, concurrency: int = DEFAULT_CONCURRENCY, headers: dict = None, timeout: float = DEFAULT_TIMEOUT):
        self.concurrency = max(1, int(concurrency))
        self.sema = asyncio.Semaphore(self.concurrency)
        self.headers = headers or DEFAULT_HEADERS
        self.timeout = timeout
        self._client = None  # will be created on first use

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(headers=self.headers, timeout=self.timeout)

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _fetch(self, url: str) -> Optional[str]:
        await self._ensure_client()
        async with self.sema:
            try:
                r = await self._client.get(url)
                if r.status_code == 200:
                    return r.text
                else:
                    # keep warnings non-fatal
                    st.warning(f"[collector] HTTP {r.status_code} for {url}")
                    return None
            except Exception as e:
                st.warning(f"[collector] fetch error: {e} for {url}")
                return None

    def _parse_search_html_lxml(self, html_text: str, max_items: int = 30) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not html_text:
            return out
        try:
            doc = lxml_html.fromstring(html_text)
            # Heuristique : ancres contenant '/item/' (adapter si structure Vinted différente)
            anchors = doc.xpath("//a[contains(@href,'/item/')]")
            seen = set()
            for a in anchors:
                if len(out) >= max_items:
                    break
                href = a.get("href")
                if not href:
                    continue
                href_full = href if href.startswith("http") else ("https://www.vinted.fr" + href)
                if href_full in seen:
                    continue
                seen.add(href_full)

                title = (a.text_content() or "").strip()
                price_el = a.xpath(".//*[contains(text(),'€')][1]")
                asking = 0.0
                if price_el:
                    txt = price_el[0].text_content()
                    m = re.search(r"(\d+[,.]?\d*)", txt)
                    if m:
                        try:
                            asking = float(m.group(1).replace(",", "."))
                        except Exception:
                            asking = 0.0

                thumb = None
                img = a.xpath(".//img[1]")
                if img:
                    thumb = img[0].get("src") or img[0].get("data-src") or img[0].get("data-lazy-src")

                out.append({
                    "title": title,
                    "asking_price": asking,
                    "url": href_full,
                    "posted_at": None,
                    "likes": 0,
                    "views": 0,
                    "thumbnails": [thumb] if thumb else [],
                    "historical_prices": []
                })
        except Exception as e:
            st.warning(f"[collector] parse error: {e}")
        return out

    async def fetch_and_parse(self, url: str, max_items: int = 30) -> List[Dict[str, Any]]:
        html = await self._fetch(url)
        if not html:
            return []
        return self._parse_search_html_lxml(html, max_items=max_items)

# -----------------------
# Enrichisseur simple
# -----------------------
def enrich_listing_with_price_history(listing: Dict[str, Any]) -> Dict[str, Any]:
    asking = listing.get("asking_price", 0.0)
    if asking and asking > 0:
        market_est = round(asking * 1.5, 2)
        listing["historical_prices"] = [market_est, market_est * 0.9, market_est * 1.1]
        listing["market_price_est"] = estimate_market_price(listing["historical_prices"])
    else:
        listing["historical_prices"] = []
        listing["market_price_est"] = None
    listing["brand_popularity"] = 0.5
    listing["photo_quality"] = 0.9 if listing.get("thumbnails") else 0.4
    listing["ambiguous_brand"] = False
    listing["suspect_low_price"] = (listing["market_price_est"] is not None) and (listing["asking_price"] < max(1.0, 0.2 * listing["market_price_est"]))
    return listing

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Vinted Scanner", layout="wide")
st.title("Vinted — Scanner Achat/Revente (prototype)")

with st.sidebar:
    st.header("Paramètres")
    url = st.text_input("URL page recherche Vinted (ex: https://www.vinted.fr/vetements?search_text=sneakers)")
    max_items = st.number_input("Max items à parser", min_value=5, max_value=200, value=30, step=5)
    top_n = st.number_input("Top N résultats à afficher", min_value=1, max_value=50, value=10)
    concurrency = st.slider("Concurrency (fetch parallel)", 1, 12, DEFAULT_CONCURRENCY)
    run_scan = st.button("Lancer le scan")
    st.markdown("---")
    st.markdown("⚠️ Respecte les Terms of Service de Vinted — n'automatise pas les achats.")

col1, col2 = st.columns([2,1])

def run_scan_and_get_results(scan_url: str, max_items: int, concurrency: int) -> List[Dict[str, Any]]:
    """Helper sync wrapper to call async collector from Streamlit."""
    collector = CollectorAsync(concurrency=concurrency)
    try:
        # create a new event loop for this call to avoid issues in some Streamlit envs
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            raw_items = loop.run_until_complete(collector.fetch_and_parse(scan_url, max_items=max_items))
        finally:
            loop.run_until_complete(collector.close())
            loop.close()
    except Exception as e:
        st.warning(f"Erreur durant le scan: {e}")
        return []

    # enrich and score
    scored_rows: List[Dict[str, Any]] = []
    for it in raw_items:
        try:
            it = enrich_listing_with_price_history(it)
            scored = score_listing(it)
            row = {
                "title": it.get("title", "")[:120],
                "asking_price": scored["asking_price"],
                "market_est": scored["market_price_est"],
                "net_profit": scored["net_profit"],
                "score": scored["score"],
                "url": it.get("url"),
                "thumbnail": it.get("thumbnails")[0] if it.get("thumbnails") else ""
            }
            scored_rows.append(row)
        except Exception as e:
            # skip problematic item but continue
            st.warning(f"Erreur item: {e}")
            continue

    # sort by score desc
    scored_rows_sorted = sorted(scored_rows, key=lambda r: r.get("score", 0), reverse=True)
    return scored_rows_sorted

if run_scan:
    if not url:
        st.sidebar.error("Renseigne une URL Vinted dans la sidebar.")
    else:
        with st.spinner("Scan en cours — récupération et scoring..."):
            t0 = time.time()
            results = run_scan_and_get_results(url, max_items=max_items, concurrency=concurrency)
            elapsed = time.time() - t0
            st.success(f"Scan terminé — {len(results)} items traité(s) en {elapsed:.2f}s")

        if not results:
            st.info("Aucun item récupéré — vérifie l'URL ou adapte les sélecteurs XPath si nécessaire.")
        else:
            df = pd.DataFrame(results[:top_n])
            # display left: table; right: thumbnails + quick list
            with col1:
                st.subheader("Top résultats")
                def make_clickable(url, title):
                    return f"[{title}]({url})" if url else title
                df_display = df.copy()
                df_display["item"] = df_display.apply(lambda r: make_clickable(r["url"], r["title"]), axis=1)
                df_display = df_display[["item", "asking_price", "market_est", "net_profit", "score"]]
                st.table(df_display)

            with col2:
                st.subheader("Sélection rapide")
                for r in results[:min(6, len(results))]:
                    st.markdown(f"**{r['title']}** — Score: **{r['score']}**")
                    if r["thumbnail"]:
                        try:
                            st.image(r["thumbnail"], width=150)
                        except Exception:
                            # image may not load; ignore silently
                            pass
                    if r.get("url"):
                        st.markdown(f"[Voir l'annonce]({r['url']})")

# non-blocking footer
st.markdown("---")
st.caption("Prototype — adapte les sélecteurs XPath selon le HTML réel de Vinted. Respecte les ToS.")
