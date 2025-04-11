from datetime import datetime
import itemadapter
import scrapy
from itemloaders import ItemLoader
from webscraper.items import RawDocument
import mimetypes
import os
import re
import urllib.parse
from dotenv import load_dotenv

class SupremeCourtSpider(scrapy.Spider):
    """
    Spider for scraping Supreme Court website dockets and amicus briefs.
    
    Usage:
        `scrapy crawl supremecourt`
    """
    name = "supremecourt"
    
    def start_requests(self):
        """
        Generates initial requests to scrape Supreme Court dockets from years 18-24.
        """
        print('Scraping Supreme Court dockets')
        
        # Uncomment to scrape all dockets from 2018 to 2024
        for year in range(18, 25):
            for docketid in range(1, 20000):
                yield scrapy.Request(
                    f"https://www.supremecourt.gov/search.aspx?filename=/docket/docketfiles/html/public/{year}-{docketid}.html",
                    callback=self.parse,
                    cb_kwargs={"doctype": "supreme_docket", "storedoc": True}
                )
    
    def parse(self, response, **kwargs):
        """
        Main parse method that handles all response types based on doctype.
        """
        print(response.url)
        
        doctype = kwargs['doctype']
        storedoc = kwargs.get('storedoc', True)
        
        if storedoc:
            # Extract and store the document
            meta_extract_dict = self._build_meta_dict(response, doctype)
            
            # Handle POST request URLs
            if 'formdata' in response.meta:
                meta_extract_dict['url'] = (
                    "value",
                    response.url + '&' + '&'.join([
                        f'{k}={v}' for k, v in response.meta['formdata'].items()
                        if (0 < len(str(v)) < 50) and (f'{k}={v}' not in response.url)
                    ]))
            
            # Create item loader and populate with metadata
            loader = ItemLoader(item=RawDocument(), response=response, selector=response)
            for key, (valtype, val) in meta_extract_dict.items():
                if valtype == 'xpath':
                    loader.add_xpath(key, val)
                elif valtype == 'css':
                    loader.add_css(key, val)
                elif valtype == 'value':
                    loader.add_value(key, val)
                elif valtype == 'func':
                    loader.add_value(key, val(response))
            
            yield loader.load_item()
        
        # Generate new requests based on doctype
        yield from self._process_by_doctype(response, doctype)
    
    def _build_meta_dict(self, response, doctype):
        """
        Build metadata dictionary for the response.
        """
        return {
            'url': ('value', response.url),
            'state': ('value', 'SupremeCourt'),
            'rawcontent': ('func', lambda r: r.body),
            'doctype': ('value', doctype),
            'accessed': ('value', datetime.now()),
            'filetype': ('func', lambda r: mimetypes.guess_extension(
                str(r.headers[b'Content-Type'].decode()).split(';')[0], strict=False
            )),
            'metadata': ('value', {})
        }
    
    def _process_by_doctype(self, response, doctype):
        """
        Process the response based on doctype and generate new requests.
        """
        if doctype == "supreme_home":
            viewstate = response.xpath('//input[@id = "__VIEWSTATE"]/@value').get()
            viewstategenerator = response.xpath('//input[@id = "__VIEWSTATEGENERATOR"]/@value').get()

            formdata = {
                "ctl00_ctl00_RadScriptManager1_TSM": "",
                "__EVENTTARGET": "",
                "__EVENTARGUMENT": "",
                "__VIEWSTATE": viewstate,
                "__VIEWSTATEGENERATOR": viewstategenerator,
                "ctl00$ctl00$txtSearch": "",
                "ctl00$ctl00$txbhidden": "",
                "ct": "Supreme-Court-Dockets",
                "ctl00$ctl00$MainEditable$mainContent$cmdSearch": "Search"
            }

            for year in range(18, 25):
                for docketnum in range(1, 20000):
                    formdata["ctl00$ctl00$MainEditable$mainContent$txtQuery"] = f"{year}-{docketnum}"
                    yield scrapy.FormRequest(
                        "https://www.supremecourt.gov/docket/docket.aspx",
                        formdata=formdata,
                        callback=self.parse,
                        cb_kwargs={"doctype": "supreme_docket_search", "storedoc": True}
                    )
        
        elif doctype == "supreme_docket_search":
            # Find links to docket files
            links = response.xpath('//a[contains(@href, "docketfiles")]/@href').getall()
            for link in links:
                yield scrapy.Request(
                    url=urllib.parse.urljoin(response.url, link),
                    callback=self.parse,
                    cb_kwargs={"doctype": "supreme_docket", "storedoc": True}
                )
        
        elif doctype == "supreme_docket":
            # Look for pages with amicus briefs and collect document links
            page_text = response.xpath('//body').get()
            if re.search(r"amicus|amici", page_text, re.IGNORECASE):
                document_links = response.xpath('//a[contains(@class, "documentanchor")]')
                
                for link in document_links:
                    url = link.xpath('@href').get()
                    yield scrapy.Request(
                        url=url,
                        callback=self.parse,
                        cb_kwargs={
                            "doctype": "supreme_document",
                            "storedoc": True,
                        }
                    )