import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
import xml.etree.ElementTree as ET
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import json
import os

# ตั้งค่า matplotlib ให้รองรับภาษาไทย
plt.rcParams['font.family'] = 'TH Sarabun New'

# ขยาย Thai Sentiment Lexicon
THAI_SENTIMENT_LEXICON = {
    # คำเชิงบวกมาก (0.8-1.0)
    "ดีเยี่ยม": 1.0, "เยี่ยมยอด": 1.0, "สุดยอด": 1.0, "ยอดเยี่ยม": 1.0, 
    "เจริญ": 0.9, "รุ่งเรือง": 0.9, "เติบโต": 0.9, "พุ่ง": 0.9, "ทะยาน": 0.9,
    "สำเร็จ": 0.8, "ชนะ": 0.8, "ดี": 0.8, "เยี่ยม": 0.9, "ยอด": 0.8,
    # คำเชิงบวกปานกลาง (0.4-0.7)
    "ชอบ": 0.7, "พอใจ": 0.7, "ยินดี": 0.7, "ดีใจ": 0.7, "สดใส": 0.7,
    "ขึ้น": 0.6, "เพิ่ม": 0.6, "ดีขึ้น": 0.6, "ฟื้นตัว": 0.6, "แข็งแกร่ง": 0.6,
    "มั่นคง": 0.5, "ราบรื่น": 0.5,
    # คำเชิงลบมาก (-0.8 ถึง -1.0)
    "แย่มาก": -1.0, "ล้มเหลว": -1.0, "เจ๊ง": -1.0, "ล่มสลาย": -1.0, "วิกฤต": -1.0,
    "ทุจริต": -0.9, "โกง": -0.9, "ฉ้อโกง": -0.9, "หลอกลวง": -0.9,
    "ขาดทุน": -0.9, "ตกต่ำ": -0.9, "ย่ำแย่": -0.9, "ดิ่ง": -0.9,
    "แย่": -0.8, "สแกม": -0.8, "สแกมเมอร์": -0.8, "เสีย": -0.8, "เลวร้าย": -0.8,
    # คำเชิงลบปานกลาง (-0.4 ถึง -0.7)
    "ปัญหา": -0.7, "กังวล": -0.7, "ห่วง": -0.7, "เสี่ยง": -0.7, "อันตราย": -0.7,
    "ลดลง": -0.6, "ลด": -0.6, "หด": -0.6, "ตก": -0.6, "ลง": -0.6,
    "อ่อนแอ": -0.5, "ชะลอ": -0.5, "ซบเซา": -0.5, "ติดขัด": -0.5,
    # เศรษฐกิจและการเงิน
    "กำไร": 0.8, "รายได้": 0.6, "ลงทุน": 0.5,
    "หนี้": -0.6, "ขาดดุล": -0.7, "เงินเฟ้อ": -0.6, "ว่างงาน": -0.7,
    # หุ้น
    "ขาย": -0.3, "ถือ": 0.1, "ซื้อ": 0.4, "แนะนำซื้อ": 0.7,
    # คำเสริม
    "มาก": 1.2, "สุด": 1.3, "ที่สุด": 1.3, "ไม่": -1.5,
}

THAI_STOPWORDS = set(thai_stopwords())

def get_google_news(keyword, lang="th", max_results=100):
    """ดึงข่าวล่าสุดจาก Google News"""
    if lang == "th":
        url = f"https://news.google.com/rss/search?q={keyword}&hl=th&gl=TH&ceid=TH:th"
    elif lang == "en":
        url = f"https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"
    else:
        print("⚠️ ภาษาที่รองรับ: 'th' หรือ 'en'")
        return []

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return []
    
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        print(f"❌ XML Error: {e}")
        return []
    
    news_list = []
    for item in root.findall('./channel/item'):
        if len(news_list) >= max_results:
            break
        title = item.find('title').text if item.find('title') is not None else 'N/A'
        link = item.find('link').text if item.find('link') is not None else 'N/A'
        pub_date = item.find('pubDate').text if item.find('pubDate') is not None else 'N/A'
        source = item.find('source').text if item.find('source') is not None else 'Google News'
        news_list.append({'keyword': keyword, 'title': title, 'link': link, 
                         'pubDate': pub_date, 'source': source})
    return news_list

def parse_news(news_list):
    """แปลงข้อมูลข่าว"""
    parsed_news = []
    for news_item in news_list:
        try:
            dt = datetime.strptime(news_item['pubDate'], '%a, %d %b %Y %H:%M:%S %Z')
            current_date = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M:%S")
        except:
            current_date = date.today().strftime("%Y-%m-%d")
            time_str = "N/A"
        parsed_news.append({
            'keyword': news_item.get('keyword', 'N/A'),
            'date': current_date, 'time': time_str,
            'title': news_item['title'],
            'source': news_item.get('source', 'N/A'),
            'link': news_item.get('link', 'N/A')
        })
    return parsed_news

def analyze_sentiment_lexicon(title):
    """วิเคราะห์ sentiment ด้วย Lexicon"""
    tokens = word_tokenize(title, engine='newmm')
    total_score, word_count = 0, 0
    matched_words, negation = [], False
    
    for i, token in enumerate(tokens):
        token = token.strip()
        if not token or len(token) < 2:
            continue
        if token in ['ไม่', 'ไม่ใช่', 'ไม่ได้', 'มิ']:
            negation = True
            continue
        if token in THAI_STOPWORDS and token not in THAI_SENTIMENT_LEXICON:
            continue
        
        score = THAI_SENTIMENT_LEXICON.get(token, 0)
        if score != 0:
            if negation:
                score = -score
                negation = False
            if i + 1 < len(tokens):
                next_token = tokens[i + 1].strip()
                intensifier = THAI_SENTIMENT_LEXICON.get(next_token, 1.0)
                if intensifier > 1.0:
                    score *= (intensifier / 1.0)
            total_score += score
            word_count += 1
            matched_words.append(f"{token}({score:.2f})")
    
    polarity = max(-1.0, min(1.0, total_score / word_count if word_count > 0 else 0))
    label = 'positive' if polarity > 0.1 else 'negative' if polarity < -0.1 else 'neutral'
    return polarity, label, matched_words

def analyze_sentiment(parsed_news):
    """วิเคราะห์ sentiment ทั้งหมด"""
    analyzed_news = []
    for news in parsed_news:
        polarity, label, matched = analyze_sentiment_lexicon(news['title'])
        news['sentiment'] = polarity
        news['sentiment_label'] = label
        news['matched_words'] = ', '.join(matched) if matched else 'ไม่มี'
        analyzed_news.append(news)
    return analyzed_news

def save_results(df, keyword, output_dir='results'):
    """บันทึกผลลัพธ์"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_keyword = keyword.replace(" ", "_").replace("/", "_")
    base = f"{output_dir}/{safe_keyword}_{timestamp}"
    
    # CSV
    try:
        df.to_csv(f"{base}.csv", index=False, encoding='utf-8-sig')
        print(f"✅ CSV: {base}.csv")
    except Exception as e:
        print(f"⚠️ CSV Error: {e}")
    
    # Excel
    try:
        df.to_excel(f"{base}.xlsx", index=False, engine='openpyxl')
        print(f"✅ Excel: {base}.xlsx")
    except Exception as e:
        print(f"⚠️ Excel Error: {e}")
    
    # JSON
    try:
        df.to_json(f"{base}.json", orient='records', force_ascii=False, indent=2)
        print(f"✅ JSON: {base}.json")
    except Exception as e:
        print(f"⚠️ JSON Error: {e}")
    
    # Summary
    try:
        with open(f"{base}_summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n📊 รายงาน Sentiment: {keyword}\n{'='*70}\n\n")
            f.write(f"จำนวนข่าว: {len(df)}\n")
            f.write(f"ค่าเฉลี่ย: {df['sentiment'].mean():.4f}\n")
            f.write(f"ค่าสูงสุด: {df['sentiment'].max():.4f}\n")
            f.write(f"ค่าต่ำสุด: {df['sentiment'].min():.4f}\n\n")
            counts = df['sentiment_label'].value_counts()
            for label, count in counts.items():
                f.write(f"{label}: {count} ({count/len(df)*100:.1f}%)\n")
            f.write(f"\n{'='*70}\nTOP 3 บวก:\n")
            for idx, row in df.nlargest(3, 'sentiment').iterrows():
                f.write(f"{row['sentiment']:.3f} - {row['title'][:100]}\n")
            f.write(f"\n{'='*70}\nTOP 3 ลบ:\n")
            for idx, row in df.nsmallest(3, 'sentiment').iterrows():
                f.write(f"{row['sentiment']:.3f} - {row['title'][:100]}\n")
        print(f"✅ Summary: {base}_summary.txt")
    except Exception as e:
        print(f"⚠️ Summary Error: {e}")

def plot_sentiment(df, ticker, avg_sentiment, save_fig=True, output_dir='results'):
    """สร้างกราฟ"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = ticker.replace(" ", "_").replace("/", "_")
    df['news_num'] = range(1, len(df) + 1)
    
    # Scatter
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = df['sentiment'].apply(lambda x: '#4CAF50' if x > 0.1 else '#F44336' if x < -0.1 else '#FFC107')
    ax.scatter(df['news_num'], df['sentiment'], alpha=0.6, s=80, c=colors, edgecolors='black')
    ax.plot(df['news_num'], df['sentiment'], alpha=0.3, color='gray')
    ax.axhline(avg_sentiment, color='blue', linestyle='--', linewidth=2, label=f'ค่าเฉลี่ย: {avg_sentiment:.3f}')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='เป็นกลาง')
    ax.set_title(f'Sentiment Analysis: {ticker}', fontsize=16, fontweight='bold')
    ax.set_xlabel('ลำดับข่าว', fontsize=12)
    ax.set_ylabel('Sentiment', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{output_dir}/{safe_ticker}_{timestamp}_scatter.png", dpi=300)
    plt.show()
    
    # Histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    n, bins, patches = ax.hist(df['sentiment'], bins=25, edgecolor='black', alpha=0.7)
    for i, patch in enumerate(patches):
        patch.set_facecolor('#4CAF50' if bins[i] > 0.1 else '#F44336' if bins[i] < -0.1 else '#FFC107')
    ax.axvline(avg_sentiment, color='blue', linestyle='--', linewidth=2, label=f'ค่าเฉลี่ย: {avg_sentiment:.3f}')
    ax.set_title(f'การกระจาย Sentiment: {ticker}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('จำนวนข่าว', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{output_dir}/{safe_ticker}_{timestamp}_histogram.png", dpi=300)
    plt.show()
    
    # Pie
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = (df['sentiment'] > 0.1).sum()
    neg = (df['sentiment'] < -0.1).sum()
    neu = len(df) - pos - neg
    wedges, texts, autotexts = ax.pie([pos, neg, neu], labels=['Positive', 'Negative', 'Neutral'],
                                       colors=['#4CAF50', '#F44336', '#FFC107'],
                                       autopct='%1.1f%%', startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title(f'สัดส่วน Sentiment: {ticker}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{output_dir}/{safe_ticker}_{timestamp}_pie.png", dpi=300)
    plt.show()

def compare_keywords(results_dict, output_dir='results'):
    """เปรียบเทียบหลายคำค้น"""
    data = []
    for kw, df in results_dict.items():
        data.append({
            'keyword': kw,
            'avg': df['sentiment'].mean(),
            'pos_pct': (df['sentiment'] > 0.1).sum() / len(df) * 100,
            'neg_pct': (df['sentiment'] < -0.1).sum() / len(df) * 100,
            'total': len(df)
        })
    comp_df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#4CAF50' if v > 0 else '#F44336' if v < 0 else '#FFC107' for v in comp_df['avg']]
    bars = ax.bar(range(len(comp_df)), comp_df['avg'], color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-')
    ax.set_title('เปรียบเทียบ Sentiment', fontsize=18, fontweight='bold')
    ax.set_xticks(range(len(comp_df)))
    ax.set_xticklabels(comp_df['keyword'], rotation=30, ha='right')
    ax.set_ylabel('ค่าเฉลี่ย Sentiment')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, comp_df['avg']):
        ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}',
               ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/comparison_{timestamp}.png", dpi=300)
    print(f"✅ กราฟเปรียบเทียบ: {output_dir}/comparison_{timestamp}.png")
    plt.show()
    
    comp_df.to_csv(f"{output_dir}/comparison_{timestamp}.csv", index=False, encoding='utf-8-sig')
    print(f"✅ ตารางเปรียบเทียบ: {output_dir}/comparison_{timestamp}.csv")
    return comp_df

def main(ticker, lang="th", max_results=100, save_files=True):
    """ฟังก์ชันหลัก"""
    print(f"\n{'='*60}\n🔍 กำลังดึงข่าว: {ticker}\n{'='*60}")
    news_list = get_google_news(ticker, lang=lang, max_results=max_results)
    if not news_list:
        print(f"❌ ไม่พบข่าวสำหรับ '{ticker}'")
        return None
    
    print(f"✅ ดึงข่าวได้: {len(news_list)} ข่าว")
    parsed = parse_news(news_list)
    analyzed = analyze_sentiment(parsed)
    df = pd.DataFrame(analyzed)
    avg_sentiment = df['sentiment'].mean()
    
    print(f"\n📊 สรุป:")
    print(f"  ค่าเฉลี่ย: {avg_sentiment:.4f}")
    counts = df['sentiment_label'].value_counts()
    for label, count in counts.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    if save_files:
        print(f"\n💾 บันทึกไฟล์...")
        save_results(df, ticker)
        plot_sentiment(df, ticker, avg_sentiment, save_fig=True)
    else:
        plot_sentiment(df, ticker, avg_sentiment, save_fig=False)
    
    return df

def analyze_multiple(keywords, lang="th", max_results=50, save_files=True):
    """วิเคราะห์หลายคำค้น"""
    results = {}
    for kw in keywords:
        df = main(kw, lang=lang, max_results=max_results, save_files=save_files)
        if df is not None:
            results[kw] = df
    
    if len(results) > 1:
        print(f"\n{'='*60}\n📊 สร้างกราฟเปรียบเทียบ\n{'='*60}")
        comp_df = compare_keywords(results)
        return results, comp_df
    return results, None

# ===== การใช้งาน =====
if __name__ == "__main__":
    print("="*60)
    print("🇹🇭 Thai Sentiment Analyzer")
    print("="*60)
    
    # ตัวเลือก 1: วิเคราะห์คำเดียว
    # ticker = input("\nใส่คำค้น: ").strip() or "หุ้นไทย"
    # main(ticker, lang="th", max_results=50, save_files=True)
    
    # ตัวเลือก 2: วิเคราะห์หลายคำ (แนะนำ)
    keywords = ["หุ้นไทย SET", "ราคาทองคำ", "อสังหาริมทรัพย์", "เศรษฐกิจไทย"]
    
    # หรือให้ผู้ใช้ใส่เอง
    # user_input = input("\nใส่คำค้น (คั่นด้วย ,): ").strip()
    # if user_input:
    #     keywords = [k.strip() for k in user_input.split(',')]
    
    results, comparison = analyze_multiple(keywords, lang="th", max_results=50, save_files=True)
    
    print("\n" + "="*60)
    print("✨ เสร็จสมบูรณ์!")
    print("="*60)
    print("📁 ไฟล์บันทึกใน folder 'results/'")
    print("  - CSV, Excel, JSON")
    print("  - Summary.txt")
    print("  - กราฟ PNG")