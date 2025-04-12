import scrapy


class RawDocument(scrapy.Item):
    # An Item class for storing raw documents with their metadata
    rawcontent = scrapy.Field()
    url = scrapy.Field()
    accessed = scrapy.Field()
    doctype = scrapy.Field()
    filetype = scrapy.Field()
    metadata = scrapy.Field()
    is_amicus_brief = scrapy.Field()
    extracted_text = scrapy.Field()
    case_title = scrapy.Field()
    docket_page = scrapy.Field()
    date = scrapy.Field()