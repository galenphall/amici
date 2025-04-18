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

    def __init__(self, *args, **kwargs):
        """
        Initialize the spider with configurable parameters.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments containing configuration parameters
        """
        super().__init__(*args, **kwargs)

        # Log configuration
        logging.basicConfig(level=logging.INFO)
        
        if kwargs.get('pdfurls') is not None:
            self.pdfurls = kwargs['pdfurls']
            self.override_start = True

            self.logger.info(f"Configured to scrape PDF URLs from: {self.pdfurls}")
        else:
            self.pdfurls = False
            self.override_start = False

            # Override default settings if provided
            self.start_year = kwargs.get('start_year', self.start_year)
            self.end_year = kwargs.get('end_year', self.end_year)
            self.min_docket = kwargs.get('min_docket', self.min_docket)
            self.max_docket = kwargs.get('max_docket', self.max_docket)
            
            self.logger.info(f"Configured to scrape dockets from {self.start_year} to {self.end_year}")
    
    def start_requests(self):
        """
        Generates initial requests to scrape Supreme Court dockets from configured years.
        """
        if not self.override_start:
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
        else:
            # Collect amicus brief pdf urls from lines in the provided file; 
            # each line should contain a single url.
            self.logger.info('Scraping Supreme Court amicus brief pdf urls')
            with open(self.pdfurls, 'r') as f:
                for line in f:
                    url = line.strip()
                    if url:
                        yield scrapy.Request(
                            url=url,
                            callback=self.parse,
                            cb_kwargs={"doctype": "supreme_document"},
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
            # Remove doctype from kwargs as it's passed explicitly
            kwargs_for_loader = kwargs.copy()
            kwargs_for_loader.pop('doctype', None) 
            loader = self._create_item_loader(response, doctype, **kwargs_for_loader)
            yield loader.load_item()
            
            # Process document links for amicus briefs if this is a docket page
            if doctype == "supreme_docket":
                yield from self._process_docket_page(response)
        except Exception as e:
            self.logger.error(f"Error processing {response.url}: {str(e)}")
    
    def _create_item_loader(self, response, doctype, **kwargs):
        """
        Create and populate an item loader for the response.
        
        Args:
            response: The HTTP response object
            doctype: The type of document being processed
            **kwargs: Additional keyword arguments containing metadata
            
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
        metadata = {}
        loader.add_value('metadata', metadata)
        
        # Add additional fields from kwargs
        if 'is_amicus_brief' in kwargs:
            loader.add_value('is_amicus_brief', kwargs['is_amicus_brief'])
            
        if 'case_title' in kwargs:
            loader.add_value('case_title', kwargs['case_title'])
            
        if 'docket_page' in kwargs:
            loader.add_value('docket_page', kwargs['docket_page'])
            
        if 'date' in kwargs:
            loader.add_value('date', kwargs['date'])
        
        return loader
    
    def _process_docket_page(self, response):
        """
        Process a Supreme Court docket page looking for amicus briefs.
        
        Args:
            response: The HTTP response object
            
        Returns:
            Generator yielding requests for document pages
        """
        # Get the case title from the docket info table
        case_title = None
        try:
            # Find the td containing "Title:", get the next td, then the span with class="title"
            # Get all text nodes within that span and join them
            title_parts = response.xpath('//table[@id="docketinfo"]//td[contains(span/text(), "Title:")]/following-sibling::td/span[@class="title"]//text()').getall()
            if title_parts:
                # Join parts (handling potential <br> tags) and strip whitespace
                case_title = ' '.join(part.strip() for part in title_parts if part.strip())
                case_title = case_title.strip()
        except Exception as e:
            self.logger.warning(f"Error extracting case title: {str(e)}")
            
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
                # Get the proceedings table
                proceeding_table = response.xpath('//table[@id="proceedings"]')
                rows = proceeding_table.xpath('.//tr')
                
                # Process rows in pairs (label row + document row)
                for i in range(1, len(rows), 2):  # Start at 1 to skip header
                    if i + 1 >= len(rows):
                        break
                        
                    label_row = rows[i]
                    link_row = rows[i + 1]
                    
                    # Get the date and label
                    date = label_row.xpath('./td[1]/text()').get()
                    if date:
                        date = date.strip()
                        
                    label = label_row.xpath('./td[2]/text()').get()
                    if label:
                        label = label.strip()
                        
                    # Check if this is an amicus brief
                    is_amicus = label and re.search(r"^Brief (amicus|amici)", label, re.IGNORECASE) is not None
                    
                    # Get the link from the <a> element with text "Main Document"
                    for a_tag in link_row.xpath('.//a'):
                        if a_tag.xpath('text()').get() == "Main Document":
                            url = a_tag.xpath('@href').get()
                            if url and url.endswith(".pdf"):
                                full_url = urllib.parse.urljoin(response.url, url)
                                
                                # Pass relevant metadata to the document request
                                yield scrapy.Request(
                                    url=full_url,
                                    callback=self.parse,
                                    cb_kwargs={
                                        "doctype": "supreme_document",
                                        "is_amicus_brief": is_amicus,
                                        "case_title": case_title,
                                        "docket_page": response.url,
                                        "date": date
                                    },
                                    errback=self.handle_error
                                )
            except Exception as e:
                self.logger.error(f"Error processing document links: {str(e)}")