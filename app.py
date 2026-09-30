import json
import re
from urllib.parse import urlparse, urlunparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
from google import genai
from google.genai import types
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AEO & GEO Content Scorer", page_icon="🎯", layout="centered"
)

# Constants
UA = "Mozilla/5.0 (compatible; AEOGEOScorer/1.0)"
AI_BOTS = [
    "GPTBot",
    "OAI-SearchBot",
    "ClaudeBot",
    "PerplexityBot",
    "Google-Extended",
    "CCBot",
]
MODEL = "gemini-2.5-flash"  # Fast, high-performance free-tier model
Q_START = (
    "what",
    "how",
    "why",
    "when",
    "who",
    "which",
    "can",
    "does",
    "is",
    "should",
    "are",
)


def ld_types(node, out):
  if isinstance(node, dict):
    t = node.get("@type")
    for x in t if isinstance(t, list) else [t]:
      if x:
        out.add(str(x))
    for v in node.values():
      ld_types(v, out)
  elif isinstance(node, list):
    for v in node:
      ld_types(v, out)


def llm_judge(title, headings, text, api_key):
  if not api_key:
    return None
  prompt = f"""You are auditing a web page for Answer Engine / Generative Engine Optimization (how likely AI assistants are to extract and cite it).
Title: {title}
Headings: {headings[:40]}
Page text (truncated):
{text[:12000]}

Score each 0-10 and give ONE specific, actionable fix (under 25 words) if the score is below 9:
- answer_clarity: does the page answer its core question directly and early, in concise extractable sentences?
- quotability: are claims self-contained, specific, and easy to lift as a citation (facts, numbers, definitions)?
- depth_specificity: original insight, data, examples, and completeness vs. generic filler?
- entity_trust: is it clear who is speaking, about what entity, with credible sourcing/expertise signals?
Respond with ONLY valid JSON matching this schema: {{"answer_clarity":{{"score":0,"fix":""}},"quotability":{{...}},"depth_specificity":{{...}},"entity_trust":{{...}}}}"""

  client = genai.Client(api_key=api_key)
  response = client.models.generate_content(
      model=MODEL,
      contents=prompt,
      config=types.GenerateContentConfig(
          response_mime_type="application/json", temperature=0.1
      ),
  )
  return json.loads(response.text)


def analyze_url(url, api_key):
  if not re.match(r"^https?://", url):
    url = "https://" + url
  p = urlparse(url)
  resp = requests.get(url, headers={"User-Agent": UA}, timeout=20)
  resp.raise_for_status()
  soup = BeautifulSoup(resp.text, "html.parser")

  cats, fixes = {}, []

  def chk(cat, earned, mx, fix):
    c = cats.setdefault(cat, [0, 0])
    c[0] += earned
    c[1] += mx
    if earned < mx:
      fixes.append(
          {"cat": cat, "issue": fix, "lost": round(mx - earned, 1)}
      )

  # --- Crawler access (10)
  rp = RobotFileParser()
  try:
    rr = requests.get(
        urlunparse((p.scheme, p.netloc, "/robots.txt", "", "", "")),
        headers={"User-Agent": UA},
        timeout=10,
    )
    rp.parse(rr.text.splitlines() if rr.ok else [])
  except Exception:
    rp.parse([])
  blocked = [b for b in AI_BOTS if not rp.can_fetch(b, url)]
  chk(
      "Crawler access",
      10 * (len(AI_BOTS) - len(blocked)) / len(AI_BOTS),
      10,
      f"robots.txt blocks AI crawlers: {', '.join(blocked)}. Allow them if you"
      " want AI engines to cite you.",
  )

  # --- Structured data (15)
  ld_raw, types_set = "", set()
  for s in soup.find_all("script", type="application/ld+json"):
    ld_raw += s.string or ""
    try:
      ld_types(json.loads(s.string or ""), types_set)
    except Exception:
      pass
  content_types = {
      "Article",
      "BlogPosting",
      "NewsArticle",
      "FAQPage",
      "HowTo",
      "Product",
      "Service",
      "Recipe",
      "Event",
  }
  chk(
      "Structured data",
      5 if types_set else 0,
      5,
      "No JSON-LD schema.org markup found. Add structured data.",
  )
  chk(
      "Structured data",
      5 if types_set & content_types else 0,
      5,
      "Add a content-type schema (Article, FAQPage, HowTo, Product, etc.).",
  )
  chk(
      "Structured data",
      3
      if types_set & {"Organization", "Person", "WebSite", "LocalBusiness"}
      else 0,
      3,
      "Add Organization/Person/WebSite schema so AI can identify the entity"
      " behind the page.",
  )
  chk(
      "Structured data",
      2 if types_set & {"FAQPage", "HowTo", "QAPage"} else 0,
      2,
      "Add FAQPage or HowTo schema where the content supports it.",
  )

  # --- Answer structure (15)
  h1s = soup.find_all("h1")
  h2s = soup.find_all("h2")
  heads = [
      h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])
  ]
  qheads = [
      h for h in heads if h.endswith("?") or h.lower().startswith(Q_START)
  ]
  chk(
      "Answer structure",
      4 if len(h1s) == 1 else 0,
      4,
      "Use exactly one H1 (found %d)." % len(h1s),
  )
  chk(
      "Answer structure",
      3 if len(h2s) >= 3 else len(h2s),
      3,
      "Break content into at least 3 H2 sections.",
  )
  chk(
      "Answer structure",
      3 if qheads else 0,
      3,
      "Add question-style headings (e.g. 'How does X work?') followed by"
      " direct answers.",
  )
  chk(
      "Answer structure",
      3 if soup.find(["ul", "ol", "table"]) else 0,
      3,
      "Add lists or tables; they're easy for AI to extract.",
  )
  title = (soup.title.string or "").strip() if soup.title else ""
  md = soup.find("meta", attrs={"name": "description"})
  mdc = (md.get("content") or "") if md else ""
  chk(
      "Answer structure",
      (1 if 10 <= len(title) <= 70 else 0)
      + (1 if 50 <= len(mdc) <= 170 else 0),
      2,
      "Write a 10-70 char title and a 50-170 char meta description.",
  )

  # --- Trust & freshness (10)
  has_author = bool(
      soup.find("meta", attrs={"name": "author"})
      or soup.find("a", rel="author")
      or soup.find(class_=re.compile("author|byline", re.I))
      or '"author"' in ld_raw
  )
  chk(
      "Trust & freshness",
      4 if has_author else 0,
      4,
      "Show a named author/byline (and mark it up in schema).",
  )
  has_date = bool(
      re.search(r'"date(Published|Modified)"', ld_raw)
      or soup.find("time")
      or soup.find("meta", property="article:published_time")
  )
  chk(
      "Trust & freshness",
      3 if has_date else 0,
      3,
      "Show and mark up published/updated dates.",
  )
  ext = [
      a
      for a in soup.find_all("a", href=re.compile(r"^https?://"))
      if urlparse(a["href"]).netloc != p.netloc
  ]
  chk(
      "Trust & freshness",
      min(3, len(ext) * 1.5),
      3,
      "Cite at least 2 external sources to back up claims.",
  )

  # --- Renderability (10)
  for t in soup(["script", "style", "nav", "footer", "noscript"]):
    t.decompose()
  text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
  words = len(text.split())
  chk(
      "Renderability",
      min(5, words / 300 * 5),
      5,
      f"Only {words} words in the raw HTML. AI crawlers often skip JavaScript;"
      " server-render key content.",
  )
  chk(
      "Renderability",
      2 if soup.find("link", rel="canonical") else 0,
      2,
      "Add a canonical link tag.",
  )
  chk(
      "Renderability",
      1 if soup.html and soup.html.get("lang") else 0,
      1,
      "Set the lang attribute on <html>.",
  )
  chk(
      "Renderability",
      2 if p.scheme == "https" else 0,
      2,
      "Serve the page over HTTPS.",
  )

  # --- Content quality (40, LLM)
  llm_used = False
  try:
    j = llm_judge(title, heads, text, api_key)
    if j:
      llm_used = True
      for k in (
          "answer_clarity",
          "quotability",
          "depth_specificity",
          "entity_trust",
      ):
        chk(
            "Content quality (AI-judged)",
            float(j[k]["score"]),
            10,
            j[k].get("fix") or f"Improve {k.replace('_', ' ')}.",
        )
  except Exception as e:
    fixes.append(
        {"cat": "Note", "issue": f"Gemini analysis failed: {str(e)}", "lost": 0}
    )

  earned = sum(c[0] for c in cats.values())
  mx = sum(c[1] for c in cats.values())
  score = round(earned / mx * 100) if mx > 0 else 0
  fixes.sort(key=lambda f: -f["lost"])
  return {
      "url": url,
      "score": score,
      "llm_used": llm_used,
      "categories": [
          {"name": k, "earned": round(v[0], 1), "max": v[1]}
          for k, v in cats.items()
      ],
      "fixes": fixes,
  }


# --- UI Layout ---
st.title("🎯 AEO / GEO Content Scorer")
st.write(
    "Evaluate how citable and optimized a web page is for AI answer engines"
    " (ChatGPT, Perplexity, Google AI Overviews)."
)

# Sidebar configuration for API Key management
with st.sidebar:
  st.header("Configuration")

  # Automatically load from Streamlit Secrets if available
  secret_key = ""
  try:
    secret_key = st.secrets.get("GEMINI_API_KEY", "")
  except Exception:
    pass

  api_key_input = st.text_input(
      "Google Gemini API Key",
      value=secret_key,
      type="password",
      help=(
          "Loaded automatically from Streamlit Secrets if configured, or paste"
          " your key here."
      ),
  )
  st.markdown("---")
  st.markdown(
      "**Scoring Breakdown:**\n- **60 pts:** Technical & structural rules"
      " (Schema, robots.txt, headings)\n- **40 pts:** AI-judged content"
      " quality (Clarity, quotability, trust)"
  )
    
# Main URL Input Form
url_input = st.text_input(
    "Target Page URL", placeholder="https://example.com/blog/article"
)

if st.button("Score Page", type="primary"):
  if not url_input:
    st.warning("Please enter a valid URL.")
  else:
    with st.spinner("Analyzing structure, markup, and content quality..."):
      try:
        result = analyze_url(url_input, api_key_input)

        # Score Display
        col1, col2 = st.columns([1, 2])
        with col1:
          st.metric(label="Overall Score", value=f"{result['score']}/100")
        with col2:
          if not result["llm_used"]:
            st.info(
                "ℹ️ Running in rule-only mode (No API key provided). Score is"
                " scaled to 100 based on technical rules."
            )
          else:
            st.success(
                "✅ Full hybrid audit complete (Rules + Gemini Content Judge)"
            )

        # Category Breakdown
        st.subheader("Category Breakdown")
        for cat in result["categories"]:
          pct = cat["earned"] / cat["max"] if cat["max"] > 0 else 0
          st.text(f"{cat['name']}: {cat['earned']}/{cat['max']}")
          st.progress(pct)

        # Prioritized Fixes
        st.subheader("Prioritized Fixes")
        if result["fixes"]:
          for fix in result["fixes"]:
            points_text = (
                f" (+{fix['lost']} pts)" if fix.get("lost", 0) > 0 else ""
            )
            st.markdown(
                f"- **{fix['issue']}** `({fix['cat']}{points_text})`"
            )
        else:
          st.markdown("🎉 Nothing major to fix!")

      except Exception as e:
        st.error(f"Error analyzing URL: {str(e)}")
