# Depression Brain

難治型憂鬱症（TRD）文獻日報 - 每日自動從 PubMed 抓取最新 TRD 相關文獻，由 Zhipu AI GLM-5-Turbo 分析整理，自動生成 HTML 報告並部署到 GitHub Pages。

## 架構

- **PubMed API** - 抓取 TRD 相關期刊的最新論文
- **Zhipu AI GLM-5-Turbo** - 分析、摘要、分類論文（fallback: GLM-4.7 → GLM-4.7-Flash）
- **GitHub Actions** - 每日台北時間 18:00 自動執行
- **GitHub Pages** - 部署靜態 HTML 報告

## 搜尋範圍

關鍵字與期刊來源定義於 `trd_journals_pubmed_templates.md`，涵蓋：
- 核心精神醫學期刊（Am J Psychiatry, JAMA Psychiatry, Lancet Psychiatry）
- 神經科學 / 生物精神醫學期刊
- 精神藥理學期刊
- 心理治療期刊

## 網站

- 📊 [Depression Brain 日報](https://u8901006.github.io/depression-brain/)
- 🏥 [李政洋身心診所](https://www.leepsyclinic.com/)
- 📬 [訂閱電子報](https://blog.leepsyclinic.com/)

## 本地開發

```bash
pip install -r scripts/requirements.txt
python scripts/fetch_papers.py --days 7 --max-papers 40 --json --output papers.json
python scripts/generate_report.py --input papers.json --output docs/trd-test.html --api-key $ZHIPU_API_KEY
```
