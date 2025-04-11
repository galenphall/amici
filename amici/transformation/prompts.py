APPENDIX_DETECTION_PROMPT = """
You are a legal assistant. Your task is to determine whether a legal brief contains an appendix with a list of amici.
Please provide the information in the following JSON format:
{
    "has_amici_appendix": <true or false>, (indicating if the brief has an appendix with a list of amici)
    "confidence": <int>, (confidence score from 1 to 10)
    "reason": "<reasoning for the determination>"
}
"""

AMICI_EXTRACTION_PROMPT = """
You are a legal assistant. Your task is to extract the dockets, amici, and lawyers from a legal brief.
Please provide the information in the following JSON format:
{
    "dockets": [
        {
            "year": <year>, (two-digit year, e.g., 23 for 2023)
            "number": <number>, (integer representing the docket number)
            "position": "<position>" (either "P" or "R", indicating support for the petitioner or respondent in the corresponding case)
        }
    ],
    "amici": [
        {
            "name": <name>", (the name of each amici organization or individual)
            "type": "<type>" (the type of amici, e.g., "individual", "organization", or "government")
        }
    ],
    "lawyers": [
        {
            "name": "<name>",
            "role": "<role>",
            "employer": "<employer>"
        }
    ]
}
"""