def scrape(url):
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.select_one('h1').text if soup.select_one('h1') else "No title"
    text = soup.select_one('p').text if soup.select_one('p') else "No text"
    link = soup.select_one('a').get('href') if soup.select_one('a') else "No link"

    return title, text, link


if __name__ == '__main__':
    url = "https://www.blognone.com/"
    title, text, link = scrape(url)  # ✅ ตอนนี้ฟังก์ชันรับ url ได้
    print(title)
    print(text)
    print(link)
