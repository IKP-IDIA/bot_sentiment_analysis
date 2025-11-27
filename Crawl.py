import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

async def main():
    browser_config = BrowserConfig(headless=True, verbose=True)
    run_config = CrawlerRunConfig(
        #content filtering
        word_count_threshold=10,
        exclude_external_links=True,
        excluded_tags=['form','header'],
        exclude_social_media_links=True,

        #content processing 
        process_iframes = True,
        remove_overlay_elements=True,
    )

    # Setting filter content and tranform to markdown 
    config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter = PruningContentFilter(
                threshold=0.5,
            ),
            options = {
                "ignore_links": True,
                "keep_images": False
                }
        ),
        word_count_threshold=20, #กรองเฉพาะย่อหน้าที่มีเนื้อหาจริง 
        #remove_js_scripts=True,   # ลบสคริปต์ JS ที่ไม่จำเป็น
    )

    # Create instance of crawler 
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://thestandard.co/spending-grows-despite-card-closures/" ,
            config=run_config
        )
        #print("Success:",result.success)
        #print("HTTP Status:", result.status_code)
        if not result.success:
            print(f"Crawl failed: {result.eroor_message}")
            print(f"Status code: {result.status_code}")

        print("\n เนิ้อหาภาษาไทย (Markdown): \n")
        print(result.markdown)
        
if __name__ == "__main__":
    asyncio.run(main())
