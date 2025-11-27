import asyncio
#from fastapi import FastAPI
import json
import requests
import uuid
from datetime import datetime
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
import re

async def main():
    # --- 1. Define the Markdown Generator ---
    # This tells the crawler to convert the cleaned HTML into Markdown
    cleaned_md_generator = DefaultMarkdownGenerator(
        content_source="cleaned_html",
        options={"ignore_links": True} # Helps remove remaining links
    )
    
    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=1,
        include_external=False,
        max_pages=1,
        score_threshold=0.0, # Changed to 0.0 to ensure the page is always processed
    )

    config = CrawlerRunConfig(
        deep_crawl_strategy=deep_crawl_strategy,
        # ✅ FIX: Include the markdown_generator here
        markdown_generator=cleaned_md_generator, 
        scraping_strategy=LXMLWebScrapingStrategy(),
        # target_elements=["#main-content"], # NOTE: This selector might still be incorrect
        excluded_tags=['form', 'header', 'footer', 'nav'],
        exclude_internal_links=True,
        exclude_social_media_links=True,
        verbose=False,
        stream=False
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(
            url="https://www.workpointtoday.com/open-ai-creator-of-chatgpt-brand-story",
            config=config
        )

        print(f"Crawled {len(results)} pages in total")

        all_data = []

        for result in results:
            # Check if content is actually present before processing
            content_preview = result.markdown[:50] + "..." if result.markdown else " (EMPTY CONTENT)"
            
            print("-" * 50)
            print(f"URL: {result.url}")
            print(f"Depth: {result.metadata.get('depth', 0)}")
            print(f"Content: {result.markdown}") # Showing preview now
            
            #(Meilisearch data preparation logic)
            page_id = str(uuid.uuid4())
            now_str = datetime.now().isoformat()

            page_data = {
                "id": "scp_" + page_id,
                "datetime": now_str,
                "url": result.url,
                "content": result.markdown
            }
            all_data.append(page_data)

    # Note: If content is still empty, the next step is to remove or correct 
    # the target_elements selector: target_elements=["#main-content"]

        # ส่งเข้า Meilisearch
        MEILI_URL = "http://10.1.0.150:7700/indexes/web_scraping/documents"
 
        HEADERS = {
            "Content-Type": "application/json"
        # "Authorization": "Bearer MASTER_KEY"
                }
        # try:
        #     response = requests.post(MEILI_URL, headers=HEADERS, json=[page_data])
        #     if response.status_code == 202:
        #         status = "success"
        #         message = f"ส่ง Request สำเร็จ!"
        #     else:
        #         status = "error"
        #         message = f"ส่งไม่สำเร็จ ({response.status_code})"
        # except Exception as e:
        #     status = "error"
        #     message = str(e)
    
        # print({
        #     "status": status,
        #     "message": message,
        #     #"json_content": page_data  # 🔹 แสดงเนื้อไฟล์ JSON
        #      })
 

if __name__ == "__main__":
    asyncio.run(main())

        #Access individual results 
        # for result in results[:1]: # Show first 3 results
        #     markdown_lines = result.markdown.split('\n')
            
        #     # Variables to store extracted content
        #     title = ""
        #     author = ""
        #     date = ""
        #     content_lines = []
            
        #     # Simple Content Extraction (Assuming title is thw first H1)
        #     for line in markdown_lines:
        #         #Find Title 
        #         if line.startswith('#') and not title:
        #             title = line.replace('#','').strip()
                
        #         #Find Author 
        #         elif line.startswith('โดย ') and "อัปเดต" in line:
        #             # Simple regex to get author name
        #             author_match = re.search(r'โดย \[(.*?)\]',line)
        #             if author_match:
        #                 author = author_match.group(1).strip()

        #         #Collect content lines 
        #         elif title and line.strip() and not line.startswith('##','###','---','!'):
        #             content_lines.append(line.strip())

        #     # Combine content 
        #     clean_content = "\n".join(content_lines)

            # print(f"URL: {result.url}")
            # print(f"TITLE: {title}")
            # print(f"AUTHOR: {author}")
            # print("-" * 50)
            #print(f"Content (Preview): {result.markdown.split('\n')}")
            #print("Markdown focused on target elements")
            #print("Links from entire page still available:", len(result.links.get("internal", []))