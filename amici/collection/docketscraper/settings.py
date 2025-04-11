import time

BOT_NAME = 'docketscraper'

SPIDER_MODULES = ['docketscraper.spiders']
NEWSPIDER_MODULE = 'docketscraper.spiders'

ITEM_PIPELINES = {'docketscraper.pipelines.GCSPipeline': 1}

LOG_LEVEL = 'INFO'
LOG_STDOUT = True
LOG_FILE = f'scrapy_logs/scrapy_output_{int(time.time())}.log'

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36'
RETRY_TIMES = 3
ROBOTSTXT_OBEY = True