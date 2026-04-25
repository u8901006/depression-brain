#!/usr/bin/env python3
"""
Generate TRD daily report HTML using Zhipu AI.
Model fallback: GLM-5-Turbo -> GLM-4.7 -> GLM-4.7-Flash
100000 max_tokens, 660s timeout, enhanced JSON fault tolerance.
"""

import json
import sys
import os
import re
import time
import argparse
from datetime import datetime, timezone, timedelta

import httpx

API_BASE = os.environ.get(
    "ZHIPU_API_BASE", "https://open.bigmodel.cn/api/coding/paas/v4"
)
MODEL_CHAIN = ["glm-5-turbo", "glm-4.7", "glm-4.7-flash"]

SYSTEM_PROMPT = (
    "你是難治型憂鬱症（Treatment-Resistant Depression, TRD）領域的資深研究員與科學傳播者。你的任務是：\n"
    "1. 從提供的醫學文獻中，篩選出最具臨床意義與研究價值的 TRD 論文\n"
    "2. 對每篇論文進行繁體中文摘要、分類、PICO 分析\n"
    "3. 評估其臨床實用性（高/中/低）\n"
    "4. 生成適合醫療專業人員閱讀的 TRD 日報\n\n"
    "輸出格式要求：\n"
    "- 語言：繁體中文（台灣用語）\n"
    "- 專業但易懂\n"
    "- 每篇論文需包含：中文標題、一句話總結、PICO分析、臨床實用性、分類標籤\n"
    "- 最後提供今日精選 TOP 3（最重要/最影響臨床實踐的論文）\n"
    "回傳格式必須是純 JSON，不要用 markdown code block 包裹。"
)


def load_papers(path: str) -> dict:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        cleaned = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            cleaned.append(line)
        text = "\n".join(cleaned).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    for _ in range(5):
        text = text.rstrip()
        if not text.endswith("}"):
            text += "}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            last = text.rfind("}")
            if last > 0:
                try:
                    return json.loads(text[: last + 1])
                except json.JSONDecodeError:
                    pass
    return None


def analyze_papers(api_key: str, papers_data: dict) -> dict | None:
    tz = timezone(timedelta(hours=8))
    date_str = papers_data.get(
        "date", datetime.now(tz).strftime("%Y-%m-%d")
    )
    paper_count = papers_data.get("count", 0)
    papers_text = json.dumps(
        papers_data.get("papers", []), ensure_ascii=False, indent=2
    )

    prompt = f"""以下是 {date_str} 從 PubMed 抓取的最新難治型憂鬱症（TRD）相關文獻（共 {paper_count} 篇）。

請進行以下分析，並以 JSON 格式回傳（不要用 markdown code block）：

{{
  "date": "{date_str}",
  "market_summary": "1-2句話總結今天 TRD 文獻的整體趨勢與亮點",
  "top_picks": [
    {{
      "rank": 1,
      "title_zh": "中文標題",
      "title_en": "English Title",
      "journal": "期刊名",
      "summary": "一句話總結（繁體中文，點出核心發現與臨床意義）",
      "pico": {{
        "population": "研究對象",
        "intervention": "介入措施",
        "comparison": "對照組",
        "outcome": "主要結果"
      }},
      "clinical_utility": "高/中/低",
      "utility_reason": "為什麼實用的一句話說明",
      "tags": ["標籤1", "標籤2"],
      "url": "原文連結",
      "emoji": "相關emoji"
    }}
  ],
  "all_papers": [
    {{
      "title_zh": "中文標題",
      "title_en": "English Title",
      "journal": "期刊名",
      "summary": "一句話總結",
      "clinical_utility": "高/中/低",
      "tags": ["標籤1"],
      "url": "連結",
      "emoji": "emoji"
    }}
  ],
  "keywords": ["關鍵字1", "關鍵字2"],
  "topic_distribution": {{
    "藥物治療": 3,
    "神經調節": 2
  }}
}}

原始文獻資料：
{papers_text}

請篩選出最重要的 TOP 5-8 篇論文放入 top_picks（按重要性排序），其餘放入 all_papers。
每篇 paper 的 tags 請從以下選擇：藥物治療、氯胺酮、電痙攣治療、rTMS、神經調節、心理治療、增效策略、生物標記、神經影像、自殺風險、復發預防、老年憂鬱、青少年憂鬱、共病症、精準醫療、機制研究、臨床試驗、真實世界數據、系統性回顧、藥物基因學、發炎指標、BDNF、麩胺酸系統、GABA系統、機器學習預測。
記住：回傳純 JSON，不要用 ```json``` 包裹。"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for model in MODEL_CHAIN:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 100000,
        }
        for attempt in range(3):
            try:
                print(
                    f"[INFO] {model} attempt {attempt + 1}...",
                    file=sys.stderr,
                )
                resp = httpx.post(
                    f"{API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=660,
                )
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    print(f"[WARN] Rate limited, wait {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                result = extract_json(text)
                if result is None:
                    print(
                        f"[WARN] JSON extraction failed attempt {attempt + 1}",
                        file=sys.stderr,
                    )
                    if attempt < 2:
                        time.sleep(5)
                    continue
                print(
                    f"[INFO] {model}: "
                    f"{len(result.get('top_picks', []))} top picks, "
                    f"{len(result.get('all_papers', []))} total",
                    file=sys.stderr,
                )
                return result
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON parse failed: {e}", file=sys.stderr)
                if attempt < 2:
                    time.sleep(5)
                continue
            except httpx.HTTPStatusError as e:
                print(
                    f"[ERROR] HTTP {e.response.status_code}: "
                    f"{e.response.text[:200]}",
                    file=sys.stderr,
                )
                if e.response.status_code == 429:
                    time.sleep(60 * (attempt + 1))
                    continue
                break
            except httpx.TimeoutException:
                print(
                    f"[WARN] Timeout attempt {attempt + 1} for {model}",
                    file=sys.stderr,
                )
                if attempt < 2:
                    time.sleep(10)
                continue
            except Exception as e:
                print(f"[ERROR] {model} failed: {e}", file=sys.stderr)
                break

    print("[ERROR] All models and attempts failed", file=sys.stderr)
    return None


def generate_html(analysis: dict) -> str:
    tz = timezone(timedelta(hours=8))
    date_str = analysis.get(
        "date", datetime.now(tz).strftime("%Y-%m-%d")
    )
    parts = date_str.split("-")
    if len(parts) == 3:
        date_display = f"{parts[0]}年{int(parts[1])}月{int(parts[2])}日"
    else:
        date_display = date_str

    summary = analysis.get("market_summary", "")
    top_picks = analysis.get("top_picks", [])
    all_papers = analysis.get("all_papers", [])
    keywords = analysis.get("keywords", [])
    topic_dist = analysis.get("topic_distribution", {})

    def _tags(tags):
        return "".join(f'<span class="tag">{t}</span>' for t in tags)

    def _util(u):
        if u == "\u9ad8":
            return "utility-high", f"{u}實用性"
        if u == "\u4e2d":
            return "utility-mid", f"{u}實用性"
        return "utility-low", f"{u}實用性"

    picks_html = ""
    for p in top_picks:
        uc, ut = _util(p.get("clinical_utility", "\u4e2d"))
        pico = p.get("pico", {})
        pico_html = ""
        if pico:
            pico_html = (
                '<div class="pico-grid">'
                f'<div class="pico-item"><span class="pico-label">P</span>'
                f'<span class="pico-text">{pico.get("population","-")}</span></div>'
                f'<div class="pico-item"><span class="pico-label">I</span>'
                f'<span class="pico-text">{pico.get("intervention","-")}</span></div>'
                f'<div class="pico-item"><span class="pico-label">C</span>'
                f'<span class="pico-text">{pico.get("comparison","-")}</span></div>'
                f'<div class="pico-item"><span class="pico-label">O</span>'
                f'<span class="pico-text">{pico.get("outcome","-")}</span></div>'
                "</div>"
            )
        picks_html += (
            '<div class="news-card featured">'
            '<div class="card-header">'
            f'<span class="rank-badge">#{p.get("rank","")}</span>'
            f'<span class="emoji-icon">{p.get("emoji","\U0001f9e0")}</span>'
            f'<span class="{uc}">{ut}</span>'
            "</div>"
            f"<h3>{p.get('title_zh', p.get('title_en',''))}</h3>"
            f'<p class="journal-source">{p.get("journal","")} '
            f"&middot; {p.get('title_en','')}</p>"
            f'<p>{p.get("summary","")}</p>'
            f"{pico_html}"
            '<div class="card-footer">'
            f'{_tags(p.get("tags",[]))}'
            f'<a href="{p.get("url","#")}" target="_blank">閱讀原文 →</a>'
            "</div></div>"
        )

    rest_html = ""
    for p in all_papers:
        uc, ut = _util(p.get("clinical_utility", "\u4e2d"))
        rest_html += (
            '<div class="news-card">'
            '<div class="card-header-row">'
            f'<span class="emoji-sm">{p.get("emoji","\U0001f4c4")}</span>'
            f'<span class="{uc} utility-sm">{ut}</span>'
            "</div>"
            f"<h3>{p.get('title_zh', p.get('title_en',''))}</h3>"
            f'<p class="journal-source">{p.get("journal","")}</p>'
            f'<p>{p.get("summary","")}</p>'
            '<div class="card-footer">'
            f'{_tags(p.get("tags",[]))}'
            f'<a href="{p.get("url","#")}" target="_blank">PubMed →</a>'
            "</div></div>"
        )

    kw_html = "".join(f'<span class="keyword">{k}</span>' for k in keywords)

    bars_html = ""
    if topic_dist:
        mx = max(topic_dist.values())
        for topic, count in topic_dist.items():
            w = int((count / mx) * 100)
            bars_html += (
                '<div class="topic-row">'
                f'<span class="topic-name">{topic}</span>'
                '<div class="topic-bar-bg">'
                f'<div class="topic-bar" style="width:{w}%"></div></div>'
                f'<span class="topic-count">{count}</span></div>'
            )

    total = len(top_picks) + len(all_papers)

    sections = ""
    if picks_html:
        sections += (
            '<div class="section">'
            '<div class="section-title">'
            '<span class="section-icon">\u2b50</span>今日精選 TOP Picks</div>'
            f"{picks_html}</div>"
        )
    if rest_html:
        sections += (
            '<div class="section">'
            '<div class="section-title">'
            '<span class="section-icon">\U0001f4da</span>其他值得關注的文獻</div>'
            f"{rest_html}</div>"
        )
    if bars_html:
        sections += (
            '<div class="topic-section section">'
            '<div class="section-title">'
            '<span class="section-icon">\U0001f4ca</span>主題分佈</div>'
            f"{bars_html}</div>"
        )
    if kw_html:
        sections += (
            '<div class="keywords-section section">'
            '<div class="section-title">'
            '<span class="section-icon">\U0001f3f7\ufe0f</span>關鍵字</div>'
            f'<div class="keywords">{kw_html}</div></div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Depression Brain &middot; TRD文獻日報 &middot; {date_display}</title>
<meta name="description" content="{date_display} 難治型憂鬱症（TRD）文獻日報，由 AI 自動彙整 PubMed 最新論文"/>
<style>
:root{{--bg:#fdf6f0;--surface:#fffaf6;--line:#e0cfc4;--text:#2a1e14;--muted:#7a6555;--accent:#c0583a;--accent-soft:#f5ddd3;--card-bg:color-mix(in srgb,var(--surface) 92%,white)}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:radial-gradient(circle at top,#f7ebe3 0,var(--bg) 55%,#eeddcf 100%);color:var(--text);font-family:"Noto Sans TC","PingFang TC","Helvetica Neue",Arial,sans-serif;min-height:100vh;overflow-x:hidden}}
.container{{position:relative;z-index:1;max-width:880px;margin:0 auto;padding:60px 32px 80px}}
header{{display:flex;align-items:center;gap:16px;margin-bottom:52px;animation:fadeDown .6s ease both}}
.logo{{width:48px;height:48px;border-radius:14px;background:linear-gradient(135deg,var(--accent),#e88a6a);display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;box-shadow:0 4px 20px rgba(192,88,58,.25)}}
.header-text h1{{font-size:22px;font-weight:700;letter-spacing:-.3px}}
.header-meta{{display:flex;gap:8px;margin-top:6px;flex-wrap:wrap;align-items:center}}
.badge{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;letter-spacing:.3px}}
.badge-date{{background:var(--accent-soft);border:1px solid var(--line);color:var(--accent)}}
.badge-count{{background:rgba(192,88,58,.06);border:1px solid var(--line);color:var(--muted)}}
.badge-source{{background:transparent;color:var(--muted);font-size:11px;padding:0 4px}}
.summary-card{{background:var(--card-bg);border:1px solid var(--line);border-radius:24px;padding:28px 32px;margin-bottom:32px;box-shadow:0 20px 60px rgba(42,30,20,.06);animation:fadeUp .5s ease .1s both}}
.summary-card h2{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.6px;color:var(--accent);margin-bottom:16px}}
.summary-text{{font-size:15px;line-height:1.8}}
.section{{margin-bottom:36px;animation:fadeUp .5s ease both}}
.section-title{{display:flex;align-items:center;gap:10px;font-size:17px;font-weight:700;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--line)}}
.section-icon{{width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0;background:var(--accent-soft)}}
.news-card{{background:var(--card-bg);border:1px solid var(--line);border-radius:24px;padding:22px 26px;margin-bottom:12px;box-shadow:0 8px 30px rgba(42,30,20,.04);transition:transform .2s,box-shadow .2s}}
.news-card:hover{{transform:translateY(-2px);box-shadow:0 12px 40px rgba(42,30,20,.08)}}
.news-card.featured{{border-left:3px solid var(--accent)}}
.card-header{{display:flex;align-items:center;gap:8px;margin-bottom:10px}}
.rank-badge{{background:var(--accent);color:#fff;font-weight:700;font-size:12px;padding:2px 8px;border-radius:6px}}
.emoji-icon{{font-size:18px}}
.card-header-row{{display:flex;align-items:center;gap:8px;margin-bottom:8px}}
.emoji-sm{{font-size:14px}}
.news-card h3{{font-size:15px;font-weight:600;margin-bottom:8px;line-height:1.5}}
.journal-source{{font-size:12px;color:var(--accent);margin-bottom:8px;opacity:.8}}
.news-card p{{font-size:13.5px;line-height:1.75;color:var(--muted)}}
.card-footer{{margin-top:12px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}}
.tag{{padding:2px 9px;background:var(--accent-soft);border-radius:999px;font-size:11px;color:var(--accent)}}
.news-card a{{font-size:12px;color:var(--accent);text-decoration:none;opacity:.7;margin-left:auto}}
.news-card a:hover{{opacity:1}}
.utility-high{{color:#2d7d46;font-size:11px;font-weight:600;padding:2px 8px;background:rgba(45,125,70,.1);border-radius:4px}}
.utility-mid{{color:#9f7a2e;font-size:11px;font-weight:600;padding:2px 8px;background:rgba(159,122,46,.1);border-radius:4px}}
.utility-low{{color:var(--muted);font-size:11px;font-weight:600;padding:2px 8px;background:rgba(90,109,130,.08);border-radius:4px}}
.utility-sm{{font-size:10px}}
.pico-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px;padding:12px;background:rgba(255,253,249,.8);border-radius:14px;border:1px solid var(--line)}}
.pico-item{{display:flex;gap:8px;align-items:baseline}}
.pico-label{{font-size:10px;font-weight:700;color:#fff;background:var(--accent);padding:2px 6px;border-radius:4px;flex-shrink:0}}
.pico-text{{font-size:12px;color:var(--muted);line-height:1.4}}
.keywords{{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}}
.keyword{{padding:5px 14px;background:var(--accent-soft);border:1px solid var(--line);border-radius:20px;font-size:12px;color:var(--accent);cursor:default;transition:background .2s}}
.keyword:hover{{background:rgba(192,88,58,.18)}}
.topic-row{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.topic-name{{font-size:13px;color:var(--muted);width:100px;flex-shrink:0;text-align:right}}
.topic-bar-bg{{flex:1;height:8px;background:var(--line);border-radius:4px;overflow:hidden}}
.topic-bar{{height:100%;background:linear-gradient(90deg,var(--accent),#e88a6a);border-radius:4px;transition:width .6s ease}}
.topic-count{{font-size:12px;color:var(--accent);width:24px}}
.clinic-banner{{margin-top:48px;animation:fadeUp .5s ease .3s both}}
.clinic-links{{display:flex;flex-direction:column;gap:12px}}
.clinic-link{{display:flex;align-items:center;gap:14px;padding:18px 24px;background:var(--card-bg);border:1px solid var(--line);border-radius:24px;text-decoration:none;color:var(--text);transition:all .2s;box-shadow:0 8px 30px rgba(42,30,20,.04)}}
.clinic-link:hover{{border-color:var(--accent);transform:translateY(-2px);box-shadow:0 12px 40px rgba(42,30,20,.08)}}
.clinic-icon{{font-size:28px;flex-shrink:0}}
.clinic-name{{font-size:15px;font-weight:700;flex:1}}
.clinic-arrow{{font-size:18px;color:var(--accent);font-weight:700}}
footer{{margin-top:32px;padding-top:22px;border-top:1px solid var(--line);font-size:11.5px;color:var(--muted);display:flex;justify-content:space-between;animation:fadeUp .5s ease .5s both}}
footer a{{color:var(--muted);text-decoration:none}}
footer a:hover{{color:var(--accent)}}
@keyframes fadeDown{{from{{opacity:0;transform:translateY(-16px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(16px)}}to{{opacity:1;transform:translateY(0)}}}}
@media(max-width:600px){{.container{{padding:36px 18px 60px}}.summary-card,.news-card{{padding:20px 18px}}.pico-grid{{grid-template-columns:1fr}}footer{{flex-direction:column;gap:6px;text-align:center}}.topic-name{{width:70px;font-size:11px}}}}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">\U0001f9e0</div>
    <div class="header-text">
      <h1>Depression Brain &middot; TRD文獻日報</h1>
      <div class="header-meta">
        <span class="badge badge-date">\U0001f4c5 {date_display}</span>
        <span class="badge badge-count">\U0001f4ca {total} 篇文獻</span>
        <span class="badge badge-source">Powered by PubMed + Zhipu AI GLM-5-Turbo</span>
      </div>
    </div>
  </header>
  <div class="summary-card">
    <h2>\U0001f4cb 今日 TRD 文獻趨勢</h2>
    <p class="summary-text">{summary}</p>
  </div>
  {sections}
  <div class="clinic-banner">
    <div class="clinic-links">
      <a href="https://www.leepsyclinic.com/" class="clinic-link" target="_blank">
        <span class="clinic-icon">\U0001f3e5</span>
        <span class="clinic-name">李政洋身心診所首頁</span>
        <span class="clinic-arrow">\u2192</span>
      </a>
      <a href="https://blog.leepsyclinic.com/" class="clinic-link" target="_blank">
        <span class="clinic-icon">\U0001f4ec</span>
        <span class="clinic-name">訂閱電子報</span>
        <span class="clinic-arrow">\u2192</span>
      </a>
    </div>
  </div>
  <footer>
    <span>資料來源：PubMed &middot; 分析模型：GLM-5-Turbo</span>
    <span><a href="https://github.com/u8901006/depression-brain">GitHub</a></span>
  </footer>
</div>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate TRD daily report")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ZHIPU_API_KEY", ""),
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "[ERROR] No API key. Set ZHIPU_API_KEY env or use --api-key",
            file=sys.stderr,
        )
        sys.exit(1)

    papers_data = load_papers(args.input)
    if not papers_data or not papers_data.get("papers"):
        print("[WARN] No papers, generating empty report", file=sys.stderr)
        tz = timezone(timedelta(hours=8))
        analysis = {
            "date": datetime.now(tz).strftime("%Y-%m-%d"),
            "market_summary": "今日 PubMed 暫無新的 TRD 文獻更新。請明天再查看。",
            "top_picks": [],
            "all_papers": [],
            "keywords": [],
            "topic_distribution": {},
        }
    else:
        analysis = analyze_papers(args.api_key, papers_data)
        if not analysis:
            print("[ERROR] Analysis failed", file=sys.stderr)
            sys.exit(1)

    html = generate_html(analysis)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] Report saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
