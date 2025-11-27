from crawl4ai import AsyncWebCrawler,  AdaptiveCrawler, AdaptiveConfig
import asyncio

async def main():
    config = AdaptiveConfig(
        confidence_threshold=0.8,
        max_pages=1,
        top_k_links=3,
        min_gain_threshold=0.1
    )

    async with AsyncWebCrawler() as crawler: 
        #Create an adaptive crawler (config is optional)
        adaptive = AdaptiveCrawler(crawler, config=config)

        #Start crawling with query 
        result = await adaptive.digest(
            start_url = "https://thestandard.co/spending-grows-despite-card-closures/",
            query = "asynce context managers"
        )

        #View statistics 
        adaptive.print_stats()

        #Get the most relevant content 
        relevant_pages = adaptive.get_relevant_content(top_k=5)
        for page in relevant_pages:
            print(f"-{page['url']} (score: {page['score']:2f})")
            print(page['markdow'])

if __name__=="__main__":
    asyncio.run(main())