import scrapy


class RawDocument(scrapy.Item):
    # An Item class for storing raw documents with their metadata
    rawcontent = scrapy.Field()
    state = scrapy.Field()
    url = scrapy.Field()
    accessed = scrapy.Field()
    doctype = scrapy.Field()
    filetype = scrapy.Field()
    metadata = scrapy.Field()