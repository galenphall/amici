from datetime import datetime
import logging
import scrapy
from itemloaders import ItemLoader
from ..items import RawDocument
import mimetypes
import re
import urllib.parse

class SupremeCourtSpider(scrapy.Spider):
    """
    Spider for scraping Supreme Court website dockets and amicus briefs.
    
    Usage:
        `scrapy crawl supremecourt`
    """
    name = "supremecourt"
    
    # Configuration parameters as class attributes
    start_year = 18
    end_year = 25
    min_docket = 1
    max_docket = 20000
    
    def start_requests(self):
        """
        Generates initial requests to scrape Supreme Court dockets from configured years.
        """
        self.logger.info('Scraping Supreme Court dockets')
        
        # Directly iterate through all docket pages
        for year in range(self.start_year, self.end_year):
            for docket_id in range(self.min_docket, self.max_docket):
                url = f"https://www.supremecourt.gov/search.aspx?filename=/docket/docketfiles/html/public/{year}-{docket_id}.html"
                yield scrapy.Request(
                    url=url,
                    callback=self.parse,
                    cb_kwargs={"doctype": "supreme_docket"},
                    errback=self.handle_error
                )
    
    def handle_error(self, failure):
        """
        Handle request errors to prevent spider crashes.
        
        Args:
            failure: The failure object containing error information
        """
        self.logger.error(f"Request failed: {failure.request.url}")
        self.logger.error(f"Error: {repr(failure.value)}")
    
    def parse(self, response, **kwargs):
        """
        Main parse method that handles all response types based on doctype.
        
        Args:
            response: The HTTP response object
            **kwargs: Additional keyword arguments
        
        Returns:
            Generator yielding items or requests
        """
        self.logger.debug(f"Processing: {response.url}")
        
        doctype = kwargs['doctype']
        
        try:
            # Extract and store the document
            loader = self._create_item_loader(response, doctype)
            yield loader.load_item()
            
            # Process document links for amicus briefs if this is a docket page
            if doctype == "supreme_docket":
                yield from self._process_docket_page(response)
        except Exception as e:
            self.logger.error(f"Error processing {response.url}: {str(e)}")
    
    def _create_item_loader(self, response, doctype):
        """
        Create and populate an item loader for the response.
        
        Args:
            response: The HTTP response object
            doctype: The type of document being processed
            
        Returns:
            A populated ItemLoader object
        """
        loader = ItemLoader(item=RawDocument(), response=response, selector=response)
        
        # Add standard metadata fields
        loader.add_value('url', response.url)
        loader.add_value('doctype', doctype)
        loader.add_value('accessed', datetime.now())
        loader.add_value('rawcontent', response.body)
        
        # Get content type
        try:
            content_type = str(response.headers[b'Content-Type'].decode()).split(';')[0]
            file_ext = mimetypes.guess_extension(content_type, strict=False)
            loader.add_value('filetype', file_ext)
        except Exception as e:
            self.logger.warning(f"Could not determine filetype: {str(e)}")
            loader.add_value('filetype', None)
        
        # Add empty metadata dictionary
        loader.add_value('metadata', {})
        
        return loader
    
    def _process_docket_page(self, response):
        """
        Process a Supreme Court docket page looking for amicus briefs.
        
        Args:
            response: The HTTP response object
            
        Returns:
            Generator yielding requests for document pages
        """
        # Check if page might contain amicus briefs (efficient check before parsing links)
        has_amicus = False
        try:
            # More efficient search - look in text content rather than full body HTML
            text_content = ' '.join(response.xpath('//text()').getall())
            has_amicus = re.search(r"amicus|amici", text_content, re.IGNORECASE) is not None
        except Exception as e:
            self.logger.warning(f"Error checking for amicus content: {str(e)}")
            # Fallback to assume it might have amicus content
            has_amicus = True
        
        if has_amicus:
            # Find document links for potential amicus briefs
            try:
                document_links = response.xpath('//a[contains(@class, "documentanchor")]')
                
                for link in document_links:
                    url = link.xpath('@href').get()
                    if url:
                        yield scrapy.Request(
                            url=urllib.parse.urljoin(response.url, url),
                            callback=self.parse,
                            cb_kwargs={"doctype": "supreme_document"},
                            errback=self.handle_error
                        )
            except Exception as e:
                self.logger.error(f"Error processing document links: {str(e)}")