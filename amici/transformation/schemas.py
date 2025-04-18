"""
JSON schemas for structured LLM responses in the amici extraction pipeline.
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
            "description": "Reasoning for the determination"
        }
    },
    "required": ["has_appendix"]
}

# Schema for amici extraction (no appendix case)
AMICI_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "dockets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "number": {"type": "integer"},
                    "position": {"type": "string", "enum": ["P", "R"]}
                },
                "required": ["year", "number", "position"]
            }
        },
        "amici": {
            "type": "array",
            "items": {"type": "string"}
        },
        "lawyers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "employer": {"type": "string"}
                },
                "required": ["name"]
            }
        }
    },
    "required": ["dockets", "amici"]
}
