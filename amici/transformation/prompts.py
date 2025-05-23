from typing import List, Optional, Literal
from pydantic import BaseModel
from pydantic import Field
from pydantic import constr

APPENDIX_DETECTION_PROMPT = """
You are a legal assistant analyzing Supreme Court amicus briefs. Determine if this brief contains an appendix listing amici curiae (organizations or individuals filing the brief).

Look for these indicators:
1. References in the table of contents to an appendix of amici or signatories
2. Phrases like "list continued in appendix" or "see appendix for complete list"
3. Statements like "X States" or "Y Organizations" on the cover with details elsewhere

Respond in this JSON format:
{
    "has_appendix": <true or False>,
    "confidence": <number from 0 to 1>,
    "reason": "<detailed reasoning>",
    "appendix_location": "<page number or section if known, otherwise null>"
}

Example 1:
Brief text: "BRIEF OF AMICI CURIAE STATES OF TEXAS, FLORIDA, AND 21 OTHER STATES IN SUPPORT OF PETITIONERS... TABLE OF CONTENTS... APPENDIX: LIST OF AMICI STATES................A-1"
Output: {"has_appendix": true, "confidence": 0.95, "reason": "Table of contents explicitly references an appendix listing amici states", "appendix_location": "A-1"}

Example 2:
Brief text: "BRIEF AMICUS CURIAE OF THE AMERICAN BAR ASSOCIATION IN SUPPORT OF RESPONDENTS... TABLE OF CONTENTS... ARGUMENT...CONCLUSION"
Output: {"has_appendix": False, "confidence": 0.9, "reason": "No reference to an appendix listing amici in the table of contents or first pages", "appendix_location": null}
"""

AMICI_EXTRACTION_PROMPT = """
You are a legal assistant extracting information from Supreme Court amicus briefs. Extract the docket numbers, amici organizations/individuals, and representing lawyers from this brief.

Important guidelines:
1. The docket number format is typically YY-NNNN (e.g., "20-1599")
2. Amici information is usually on the cover page and first few pages, but may also be in the appendix
3. For position, "P" means supporting petitioner(s) and "R" means supporting respondent(s)
4. Extract ALL named amici - if there's a phrase like "and 10 other organizations", find those organizations
5. For lawyers, focus on counsel of record and those with specific titles or firms listed
6. For lawyers, always include name, role, and employer - if any information is not available, use empty string ""

Respond in this JSON format:
{
    "dockets": [
        {
            "year": <two-digit year integer>,
            "number": <docket number integer>,
            "position": "<P or R>"
        }
    ],
    "amici": [
        {
            "name": "<full official name>",
            "category": "<individual|organization|government|academic|coalition>"
        }
    ],
    "all_amici_on_cover_page": <true or False>,
    "lawyers": [
        {
            "name": "<full name>",
            "role": "<role or title if specified, or empty string>",
            "employer": "<firm or organization if specified, or empty string>"
        }
    ],
    "counsel_of_record": "<name of primary counsel>"
}

Example:
Brief text: "No. 20-1599... BRIEF FOR AMICI CURIAE NATIONAL CRIME VICTIM LAW INSTITUTE IN SUPPORT OF PETITIONERS... PAUL G. CASSELL, Counsel of Record, UTAH APPELLATE PROJECT, S.J. QUINNEY COLLEGE OF LAW AT THE UNIVERSITY OF UTAH*..."
Output: {"dockets":[{"year":20,"number":1599,"position":"P"}],"amici":[{"name":"National Crime Victim Law Institute","category":"organization"}],"all_amici_on_cover_page":true,"lawyers":[{"name":"Paul G. Cassell","role":"Counsel of Record","employer":"Utah Appellate Project, S.J. Quinney College of Law at the University of Utah"}],"counsel_of_record":"Paul G. Cassell"}
"""

# Schema for appendix detection (first stage)
APPENDIX_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "has_appendix": {
            "type": "boolean",
            "description": "Whether the brief has an appendix containing a list of amici"
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score from 0 to 1"
        },
        "reason": {
            "type": "string",
            "description": "Detailed reasoning for the determination"
        },
        "appendix_location": {
            "type": ["string", "null"],
            "description": "Page number or section where appendix is located, if known"
        }
    },
    "required": ["has_appendix", "confidence", "reason"]
}

AMICI_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "dockets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "Two-digit year of docket"},
                    "number": {"type": "integer", "description": "Docket number"},
                    "position": {"type": "string", "enum": ["P", "R"], "description": "P for petitioner, R for respondent"}
                },
                "required": ["year", "number", "position"],
                "additionalProperties": False
            }
        },
        "amici": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full official name of amicus"},
                    "category": {"type": "string", "enum": ["individual", "organization", "government", "academic", "coalition"], "description": "Category of amicus"}
                },
                "required": ["name", "category"],
                "additionalProperties": False
            }
        },
        "all_amici_on_cover_page": {
            "type": "boolean",
            "description": "Whether the amici are all listed on the cover page or if some are only in the appendix or elsewhere"
        },
        "lawyers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of lawyer"},
                    "role": {"type": "string", "description": "Role or title of lawyer"},
                    "employer": {"type": "string", "description": "Firm or organization employing the lawyer"}
                },
                "required": ["name", "role", "employer"],
                "additionalProperties": False
            }
        },
        "counsel_of_record": {
            "type": "string", 
            "description": "Name of the primary counsel of record"
        }
    },
    "required": ["dockets", "amici", "all_amici_on_cover_page", "lawyers", "counsel_of_record"],
    "additionalProperties": False
}

class AppendixDetectionModel(BaseModel):
    has_appendix: bool
    confidence: float
    reason: str
    appendix_location: Optional[str] = None

class Docket(BaseModel):
    year: int
    number: int
    position: Literal["P", "R"]

class Amicus(BaseModel):
    name: str
    category: Literal["individual", "organization", "government", "academic", "coalition"]

class Lawyer(BaseModel):
    name: str
    role: str
    employer: str

class AmiciExtractionModel(BaseModel):
    dockets: List[Docket]
    amici: List[Amicus]
    all_amici_on_cover_page: bool
    complete_amici_list: bool
    lawyers: List[Lawyer]
    counsel_of_record: Optional[str] = None