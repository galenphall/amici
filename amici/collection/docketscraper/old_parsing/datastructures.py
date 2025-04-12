
import json
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

# Define Pydantic models for structured output
class Amicus(BaseModel):
    name: str
    type: Literal["organization", "individual"]

class Lawyer(BaseModel):
    name: str
    role: str = Field(description="Role such as 'Counsel of Record' or 'Attorney'")
    organization: str = Field(description="Law firm or organization name")
    representing: List[str] | Literal["all"] = Field(
        description="List of specific amici represented or 'all' if representing all amici"
    )

class Docket(BaseModel):
    year: Literal[17, 18, 19, 20, 21, 22, 23, 24, 25] = Field(description="The year of the docket number")
    number: int = Field(description="The number of the docket")

class Brief(BaseModel):
    position: Literal["support petitioner", "support respondent", "neutral/other"]
    dockets: List[str] = Field(description="List of dockets for which the amici are submitting the brief")

class ExtractionResult(BaseModel):
    amici: List[Amicus] = Field(default_factory=list)
    lawyers: List[Lawyer] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = Field(
        default="high", 
        description="Confidence level of the extraction"
    )

class ProcessedResult(BaseModel):
    source_file: str
    extraction_time: str
    amici: List[Amicus] = Field(default_factory=list)
    lawyers: List[Lawyer] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low", "error"] = "high"
    brief: Brief = Field(default_factory=dict)
    error: Optional[str] = None

json_schema = {
        "name": "extraction_result",
        "schema": {
            "type": "object",
            "properties": {
                "amici": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": ["organization", "individual"]}
                        },
                        "required": ["name", "type"],
                        "additionalProperties": False
                    }
                },
                "lawyers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                            "organization": {"type": "string"},
                            "representing": {
                                "anyOf": [
                                    {"type": "array", "items": {"type": "string"}},
                                    {"type": "string", "enum": ["all"]}
                                ]
                            }
                        },
                        "required": ["name", "role", "organization", "representing"],
                        "additionalProperties": False
                    }
                },
                "brief": {
                    "type": "object",
                    "properties": {
                        "position": {"type": "string", "enum": ["support petitioner", "support respondent", "neutral/other"]},
                        "dockets": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "year": {"type": "integer", "enum": [17, 18, 19, 20, 21, 22, 23, 24, 25]},
                                    "number": {"type": "integer"}
                                },
                            }
                        }
                    },
                    "required": ["position", "dockets"],
                    "additionalProperties": False
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"]
                }
            },
            "required": ["amici", "lawyers", "brief", "confidence"],
            "additionalProperties": False
        },
        "strict": True
    }