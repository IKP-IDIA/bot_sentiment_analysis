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
from collections import Counter

# ตั้งค่า matplotlib ให้รองรับภาษาไทย
plt.rcParams['font.family'] = 'TH Sarabun New'  # หรือ 'Tahoma'

# ขยาย Thai Sentiment Lexicon ให้ครอบคลุมมากขึ้น
THAI_SENTIMENT_LEXICON = {
    # คำเชิงบวกมาก (0.8 - 1.0)
    "ดีเยี่ยม": 1.0, "เยี่ยมยอด": 1.0, "สุดยอด": 1.0, "ยอดเยี่ยม": 1.0, 
    "เจริญ": 0.9, "รุ่งเรือง": 0.9, "เติบโต": 0.9, "พุ่ง": 0.9, "ทะยาน": 0.9,
    "สำเร็จ": 0.8, "ชนะ": 0.8, "ได้": 0.8, "ดี": 0.8, "เยี่ยม": 0.9,
    
    # คำเชิงบวกปานกลาง (0.4 - 0.7)
    "ชอบ": 0.7, "พอใจ": 0.7, "ยินดี": 0.7, "ดีใจ": 0.7, "สดใส": 0.7,
    "ขึ้น": 0.6, "เพิ่ม": 0.6, "ดีขึ้น": 0.6, "ฟื้นตัว": 0.6, "แข็งแกร่ง": 0.6,
    "มั่นคง": 0.5, "ราบรื่น": 0.5, "ปกติ": 0.4, "โอเค": 0.4,
    
    # คำเชิงลบมาก (-0.8 ถึง -1.0)
    "แย่มาก": -1.0, "ล้มเหลว": -1.0, "เจ๊ง": -1.0, "ล่มสลาย": -1.0, "วิกฤต": -1.0,
    "ทุจริต": -0.9, "โกง": -0.9, "ฉ้อโกง": -0.9, "คอร์รัปชั่น": -0.9, "หลอกลวง": -0.9,
    "ขาดทุน": -0.9, "ตกต่ำ": -0.9, "ย่ำแย่": -0.9, "ตกกระป๋อง": -0.9, "ดิ่ง": -0.9,
    "แย่": -0.8, "สแกม": -0.8, "สแกมเมอร์": -0.8, "เสีย": -0.8, "เลวร้าย": -0.8,
    
    # คำเชิงลบปานกลาง (-0.4 ถึง -0.7)
    "ปัญหา": -0.7, "กังวล": -0.7, "ห่วง": -0.7, "เสี่ยง": -0.7, "อันตราย": -0.7,
    "ลดลง": -0.6, "ลด": -0.6, "หด": -0.6, "ตก": -0.6, "ลง": -0.6,
    "อ่อนแอ": -0.5, "ชะลอ": -0.5, "ซบเซา": -0.5, "ซึม": -0.5, "ติดขัด": -0.5,
    "แพง": -0.4, "เหนื่อย": -0.4, "ยาก": -0.4,
    
    # คำเกี่ยวกับเศรษฐกิจและการเงิน
    "กำไร": 0.8, "รายได้": 0.6, "เงินทุน": 0.5, "ลงทุน": 0.5,
    "หนี้": -0.6, "ขาดดุล": -0.7, "เงินเฟ้อ": -0.6, "ว่างงาน": -0.7,
    
    # คำเกี่ยวกับหุ้น
    "แกว่ง": 0.0, "คาดการณ์": 0.0, "ประเมิน": 0.0, "วิเคราะห์": 0.0,
    "ขาย": -0.3, "ถือ": 0.1, "ซื้อ": 0.4, "แนะนำซื้อ": 0.7,
    
    # คำเสริมความหมาย (Intensifiers)
    "มาก": 1.2, "มากมาย": 1.2, "สุด": 1.3, "ที่สุด": 1.3, "เกินไป": 1.2,
    "ไม่": -1.5, "ไม่ใช่": -1.5, "ไม่ได้": -1.5,
}

THAI_STOPWORDS = set(thai_stopwords())

def get_google_news(keyword, lang="th", max_results=100):
    """
    ดึงข่าวล่าสุดจาก Google News
    """
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
        print(f"❌ Error fetching data: {e}")
        return []
    
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        print(f"❌ Error parsing XML: {e}")
        return []
    
    news_list = []
    
    for item in root.findall('./channel/item'):
        if len(news_list) >= max_results:
            break
            
        title = item.find('title').text if item.find('title') is not None else 'N/A'
        link = item.find('link').text if item.find('link') is not None else 'N/A'
        pub_date = item.find('pubDate').text if item.find('pubDate') is not None else 'N/A'
        source = item.find('source').text if item.find('source') is not None else 'Google News'

        news_list.append({
            'keyword': keyword,
            'title': title,
            'link': link,
            'pubDate': pub_date,
            'source': source
        })
    
    return news_list

def parse_news(news_list):
    """
    แปลงข้อมูลข่าวให้อยู่ในรูปแบบที่ใช้งานง่าย
    """
    parsed_news = []
    
    for news_item in news_list:
        title = news_item['title']
        pub_date_str = news_item['pubDate']
        keyword = news_item.get('keyword', 'N/A')
        source = news_item.get('source', 'N/A')
        link = news_item.get('link', 'N/A')

        try:
            dt = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %Z')
            current_date = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M:%S")
        except ValueError:
            current_date = date.today().strftime("%Y-%m-%d")
            time_str = "N/A"

        parsed_news.append({
            'keyword': keyword,
            'date': current_date,
            'time': time_str,
            'title': title,
            'source': source,
            'link': link
        })

    return parsed_news

def analyze_sentiment_lexicon(title):
    """
    วิเคราะห์ sentiment ด้วย Lexicon-based approach (ปรับปรุงแล้ว)
    """
    # 1. Tokenization
    tokens = word_tokenize(title, engine='newmm')
    
    # 2. คำนวณคะแนน
    total_score = 0
    word_count = 0
    matched_words = []
    
    # ตรวจสอบ negation (ไม่, ไม่ใช่, ไม่ได้)
    negation = False
    negation_words = ['ไม่', 'ไม่ใช่', 'ไม่ได้', 'มิ', 'มิใช่']
    
    for i, token in enumerate(tokens):
        token = token.strip()
        
        # ข้าม stopwords และคำสั้นเกินไป
        if not token or len(token) < 2:
            continue
        
        # เช็ค negation
        if token in negation_words:
            negation = True
            continue
        
        # ข้าม stopwords ทั่วไป (แต่ไม่ข้าม sentiment words)
        if token in THAI_STOPWORDS and token not in THAI_SENTIMENT_LEXICON:
            continue
        
        # หาคะแนนจาก lexicon
        score = THAI_SENTIMENT_LEXICON.get(token, 0)
        
        if score != 0:
            # ถ้าเจอคำปฏิเสธก่อนหน้า ให้กลับเครื่องหมาย
            if negation:
                score = -score
                negation = False
            
            # เช็คคำเสริมความหมาย (intensifiers) ข้างหลัง
            if i + 1 < len(tokens):
                next_token = tokens[i + 1].strip()
                intensifier = THAI_SENTIMENT_LEXICON.get(next_token, 1.0)
                if intensifier > 1.0:  # เป็น intensifier
                    score = score * (intensifier / 1.0)
            
            total_score += score
            word_count += 1
            matched_words.append(f"{token}({score:.2f})")
    
    # คำนวณ polarity เฉลี่ย
    polarity = total_score / word_count if word_count > 0 else 0
    
    # จำกัดช่วงคะแนน
    polarity = max(-1.0, min(1.0, polarity))
    
    # กำหนด label
    if polarity > 0.1:
        label = 'positive'
    elif polarity < -0.1:
        label = 'negative'
    else:
        label = 'neutral'
    
    return polarity, label, matched_words

def analyze_sentiment(parsed_news):
    """
    วิเคราะห์ sentiment ของข่าวทั้งหมด
    """
    analyzed_news = []
    
    for news in parsed_news:
        title = news['title']
        polarity, label, matched_words = analyze_sentiment_lexicon(title)
        
        news['sentiment'] = polarity
        news['sentiment_label'] = label
        news['matched_words'] = ', '.join(matched_words) if matched_words else 'ไม่มี'
        
        analyzed_news.append(news)
    
    return analyzed_news

def save_results(df, keyword, output_dir='results'):
    """
    บันทึกผลลัพธ์ในหลายรูปแบบ
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_keyword = keyword.replace(" ", "_").replace("/", "_")
    base_filename = f"{output_dir}/{safe_keyword}_{timestamp}"
    
    saved_files = {}
    
    # 1. CSV (UTF-8 with BOM for Excel compatibility)
    try:
        csv_file = f"{base_filename}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        saved_files['csv'] = csv_file
        print(f"✅ บันทึก CSV: {csv_file}")
    except Exception as e:
        print(f"⚠️ ไม่สามารถบันทึก CSV: {e}")
    
    # 2. Excel
    try:
        excel_file = f"{base_filename}.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Analysis')
        saved_files['excel'] = excel_file
        print(f"✅ บันทึก Excel: {excel_file}")
    except Exception as e:
        print(f"⚠️ ไม่สามารถบันทึก Excel: {e}")
    
    # 3. JSON
    try:
        json_file = f"{base_filename}.json"
        df.to_json(json_file, orient='records', force_ascii=False, indent=2)
        saved_files['json'] = json_file
        print(f"✅ บันทึก JSON: {json_file}")
    except Exception as e:
        print(f"⚠️ ไม่สามารถบันทึก JSON: {e}")
    
    # 4. Summary Report
    try:
        summary_file = f"{base_filename}_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"📊 รายงานการวิเคราะห์ Sentiment: {keyword}\n")
            f.write(f"{'='*70}\n\n")
            
            f.write(f"📅 วันที่วิเคราะห์: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"📰 จำนวนข่าวทั้งหมด: {len(df)} ข่าว\n\n")
            
            f.write(f"{'='*70}\n")
            f.write("📈 สถิติ Sentiment\n")
            f.write(f"{'='*70}\n")
            f.write(f"ค่าเฉลี่ย:     {df['sentiment'].mean():.4f}\n")
            f.write(f"ค่ามัธยฐาน:   {df['sentiment'].median():.4f}\n")
            f.write(f"ค่าสูงสุด:    {df['sentiment'].max():.4f}\n")
            f.write(f"ค่าต่ำสุด:    {df['sentiment'].min():.4f}\n")
            f.write(f"ส่วนเบี่ยงเบนมาตรฐาน: {df['sentiment'].std():.4f}\n\n")
            
            f.write(f"{'='*70}\n")
            f.write("📊 การกระจาย Sentiment\n")
            f.write(f"{'='*70}\n")
            sentiment_counts = df['sentiment_label'].value_counts()
            for label, count in sentiment_counts.items():
                pct = (count / len(df)) * 100
                emoji = '🟢' if label == 'positive' else '🔴' if label == 'negative' else '⚪'
                f.write(f"{emoji} {label.upper():12s}: {count:3d} ข่าว ({pct:5.1f}%)\n")
            
            f.write(f"\n{'='*70}\n")
            f.write("🟢 TOP 5 ข่าวที่มี Sentiment เชิงบวกที่สุด\n")
            f.write(f"{'='*70}\n")
            top_positive = df.nlargest(5, 'sentiment')
            for idx, (i, row) in enumerate(top_positive.iterrows(), 1):
                f.write(f"\n{idx}. [{row['sentiment']:.3f}] {row['title'][:150]}\n")
                f.write(f"   📅 {row['date']} | 🏢 {row['source']}\n")
                f.write(f"   🔤 คำที่ตรงกับ: {row['matched_words']}\n")
            
            f.write(f"\n{'='*70}\n")
            f.write("🔴 TOP 5 ข่าวที่มี Sentiment เชิงลบที่สุด\n")
            f.write(f"{'='*70}\n")
            top_negative = df.nsmallest(5, 'sentiment')
            for idx, (i, row) in enumerate(top_negative.iterrows(), 1):
                f.write(f"\n{idx}. [{row['sentiment']:.3f}] {row['title'][:150]}\n")
                f.write(f"   📅 {row['date']} | 🏢 {row['source']}\n")
                f.write(f"   🔤 คำที่ตรงกับ: {row['matched_words']}\n")
            
            f.write(f"\n{'='*70}\n")
            f.write("📰 แหล่งข่าวที่พบบ่อยที่สุด\n")
            f.write(f"{'='*70}\n")
            source_counts = df['source'].value_counts().head(10)
            for source, count in source_counts.items():
                f.write(f"  • {source}: {count} ข่าว\n")
        
        saved_files['summary'] = summary_file
        print(f"✅ บันทึก Summary: {summary_file}")
    except Exception as e:
        print(f"⚠️ ไม่สามารถบันทึก Summary: {e}")
    
    return saved_files

def plot_sentiment(df, ticker, avg_sentiment, save_fig=True, output_dir='results'):
    """
    สร้างกราฟแสดงผลการวิเคราะห์
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = ticker.replace(" ", "_").replace("/", "_")
    
    # กราฟที่ 1: Scatter plot
    fig, ax = plt.subplots(figsize=(14, 6))
    df['news_num'] = range(1, len(df) + 1)
    
    colors = df['sentiment'].apply(lambda x: '#4CAF50' if x > 0.1 else '#F44336' if x < -0.1 else '#FFC107')
    ax.scatter(df['news_num'], df['sentiment'], alpha=0.6, s=80, c=colors, edgecolors='black', linewidth=0.5)
    ax.plot(df['news_num'], df['sentiment'], alpha=0.3, linestyle='-', color='gray')
    ax.axhline(y=avg_sentiment, color='blue', linestyle='--', linewidth=2, 
               label=f'ค่าเฉลี่ย: {avg_sentiment:.3f}')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='เป็นกลาง (0)')
    
    ax.set_title(f'การวิเคราะห์ Sentiment: {ticker}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('ลำดับข่าว (ตามเวลา)', fontsize=12)
    ax.set_ylabel('คะแนน Sentiment', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Annotate outliers
    for idx, row in df.iterrows():
        if abs(row['sentiment']) > 0.7:
            ax.annotate(f"{row['news_num']}", 
                       (row['news_num'], row['sentiment']),
                       textcoords="offset points", 
                       xytext=(0, 10 if row['sentiment'] > 0 else -15),
                       ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    if save_fig:
        fig_file = f"{output_dir}/{safe_ticker}_{timestamp}_scatter.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"✅ บันทึกกราฟ: {fig_file}")
    plt.show()
    
    # กราฟที่ 2: Histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    n, bins, patches = ax.hist(df['sentiment'], bins=25, edgecolor='black', alpha=0.7)
    
    # ระบายสีตาม sentiment
    for i, patch in enumerate(patches):
        if bins[i] > 0.1:
            patch.set_facecolor('#4CAF50')
        elif bins[i] < -0.1:
            patch.set_facecolor('#F44336')
        else:
            patch.set_facecolor('#FFC107')
    
    ax.axvline(x=avg_sentiment, color='blue', linestyle='--', linewidth=2, 
               label=f'ค่าเฉลี่ย: {avg_sentiment:.3f}')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_title(f'การกระจายคะแนน Sentiment: {ticker}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('คะแนน Sentiment', fontsize=12)
    ax.set_ylabel('จำนวนข่าว', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_fig:
        fig_file = f"{output_dir}/{safe_ticker}_{timestamp}_histogram.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"✅ บันทึกกราฟ: {fig_file}")
    plt.show()
    
    # กราฟที่ 3: Pie Chart
    fig, ax = plt.subplots(figsize=(10, 8))
    positive = (df['sentiment'] > 0.1).sum()
    negative = (df['sentiment'] < -0.1).sum()
    neutral = ((df['sentiment'] >= -0.1) & (df['sentiment'] <= 0.1)).sum()
    
    sizes = [positive, negative, neutral]
    labels = ['Positive', 'Negative', 'Neutral']
    colors_pie = ['#4CAF50', '#F44336', '#FFC107']
    explode = (0.05, 0.05, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax.set_title(f'สัดส่วน Sentiment: {ticker}\n(ทั้งหมด {len(df)} ข่าว)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    if save_fig:
        fig_file = f"{output_dir}/{safe_ticker}_{timestamp}_pie.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"✅ บันทึกกราฟ: {fig_file}")
    plt.show()

def compare_multiple_keywords(results_dict, output_dir='results'):
    """
    เปรียบเทียบผลลัพธ์จากหลายคำค้น
    """
    comparison_data = []
    
    for keyword, df in results_dict.items():
        avg_sent = df['sentiment'].mean()
        pos_pct = ((df['sentiment'] > 0.1).sum() / len(df)) * 100
        neg_pct = ((df['sentiment'] < -0.1).sum() / len(df)) * 100
        neu_pct = 100 - pos_pct - neg_pct
        
        comparison_data.append({
            'keyword': keyword,
            'avg_sentiment': avg_sent,
            'positive_pct': pos_pct,
            'negative_pct': neg_pct,
            'neutral_pct': neu_pct,
            'total_news': len(df),
            'max_sentiment': df['sentiment'].max(),
            'min_sentiment': df['sentiment'].min()
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # กราฟเปรียบเทียบ: Bar Chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(len(comp_df))
    colors_bar = ['#4CAF50' if val > 0 else '#F44336' if val < 0 else '#FFC107' 
                  for val in comp_df['avg_sentiment']]
    
    bars = ax.bar(x, comp_df['avg_sentiment'], color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_title('เปรียบเทียบค่าเฉลี่ย Sentiment ของแต่ละคำค้น', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('คำค้น', fontsize=13)
    ax.set_ylabel('ค่าเฉลี่ย Sentiment', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df['keyword'], rotation=30, ha='right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # แสดงค่าบน bar
    for bar, val in zip(bars, comp_df['avg_sentiment']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}',
               ha='center', va='bottom' if height > 0 else 'top', 
               fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_file = f"{output_dir}/comparison_{timestamp}_bar.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"✅ บันทึกกราฟเปรียบเทียบ: {fig_file}")
    plt.show()
    
    # บันทึกตารางเปรียบเทียบ
    comp_file = f"{output_dir}/comparison_{timestamp}.csv"
    comp_df.to