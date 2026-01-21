from ai_alert import NEWS_API_KEY

def fetch_news(limit=5):
    """
    ดึงข่าวล่าสุดที่เกี่ยวกับทองคำ / ดอลลาร์ / FED
    """
    url = (
        "https://newsapi.org/v2/everything?"
        "q=gold OR XAU OR dollar OR FED&language=en&sortBy=publishedAt"
        f"&pageSize={limit}&apiKey={NEWS_API_KEY}"
    )
    resp = requests.get(url)
    data = resp.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"NEWSAPI error: {data}")

    articles = [
        {
            "title": a["title"],
            "description": a["description"],
            "url": a["url"],
            "publishedAt": a["publishedAt"],
        }
        for a in data["articles"]
    ]
    return articles
