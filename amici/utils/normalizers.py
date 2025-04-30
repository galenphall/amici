"""
Functions for normalizing the names of the interest groups that appear
in the Amici database. 
"""
import re
from typing import List, Optional
import editdistance
import unittest
from hypothesis import given, strategies as st

def normalize_interest_group_name(name: str) -> str:
    """
    Normalize the name of an interest group by removing common prefixes
    and suffixes, and converting to lowercase.

    Args:
        name (str): The name of the interest group.

    Returns:
        str: The normalized name.
    """
    # Remove common prefixes and suffixes
    name = re.sub(r'^(?:the|a|an)\s+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(?:inc|llc|corp|ltd|plc|gmbh|ag|sa)\b', '', name, flags=re.IGNORECASE)

    # Remove acronyms in parentheses appearing at the end of the name
    name = re.sub(r'\s*\([A-Z]{2,}\)\s*$', '', name)

    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    # Replace all types of dashes with one dash
    name = re.sub(r'\s*[-–—]\s*', '-', name)
    name = re.sub(r'[-–—]', '-', name)

    # Remove commas
    name = re.sub(r',', '', name)

    # Convert to lowercase
    name = name.lower()

    # Shorten common terms
    name = shorten_common_terms(name)

    return name

class TestNormalizeIdempotence(unittest.TestCase):
    """Test the idempotence property of normalize_interest_group_name function."""
    
    @given(st.text())
    def test_idempotence_property(self, input_string):
        """
        Test that f(f(x)) = f(x) for any string input using property-based testing.
        This will generate hundreds of random strings to test the property.
        """
        # Apply function once
        first_result = normalize_interest_group_name(input_string)
        
        # Apply function to the result
        second_result = normalize_interest_group_name(first_result)
        
        # Check that applying the function twice gives the same result as applying it once
        self.assertEqual(first_result, second_result, 
                        f"Failed idempotence for '{input_string}'. "
                        f"First result: '{first_result}', "
                        f"Second result: '{second_result}'")

    @given(st.text(alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll'),  # Uppercase and lowercase letters
        whitelist_characters=' ,.-()' + ''.join(['0123456789'])
    )))
    def test_idempotence_with_realistic_names(self, input_string):
        """
        Test idempotence with more realistic organization name inputs,
        containing letters, numbers, spaces, commas, periods, dashes and parentheses.
        """
        if input_string.strip():  # Skip empty strings after stripping
            # Apply function once
            first_result = normalize_interest_group_name(input_string)
            
            # Apply function to the result
            second_result = normalize_interest_group_name(first_result)
            
            # Check that applying the function twice gives the same result
            self.assertEqual(first_result, second_result)

def shorten_common_terms(name: str) -> str:
    """
    Shorten common terms in the name of an interest group.

    Args:
        name (str): The name of the interest group.

    Returns:
        str: The shortened name.
    """
    # Define common terms to shorten
    common_terms = {
        'association': 'assn',
        'committee': 'comm',
        'foundation': 'fdn',
        'institute': 'inst',
        'organization': 'org',
        'society': 'soc',
        'federation': 'fed',
        'international': 'intl',
        'national': 'natl',
        'state': 'st',
        'local': 'loc',
        'chapter': 'ch',
        'division': 'div',
        'group': 'grp',
        'alliance': 'all',
        'coalition': 'co',
        'network': 'net',
        'task force': 'tf',
        'working group': 'wg',
        'council': 'coun',
        'american': 'am',
        'center': 'ctr'
    }

    # Shorten common terms
    for term, abbreviation in common_terms.items():
        name = re.sub(r'\b' + term + r'\b', abbreviation + '.', name, flags=re.IGNORECASE)

    # Handle misspellings using editdistance package    
    def edit_distance_at_most_one(s1, s2):
        """
        Check if two strings differ by at most one edit.
        Uses the editdistance package to calculate Levenshtein distance.
        """
        return editdistance.eval(s1, s2) <= 1

    # Handle single-word term misspellings
    words = name.split()
    for i, word in enumerate(words):
        # Skip if the word is already an abbreviation
        if word.lower() in [abbr.lower() for abbr in common_terms.values()]:
            continue
            
        for term, abbreviation in common_terms.items():
            # Skip multi-word terms and exact matches (already handled)
            if ' ' not in term and word.lower() != term.lower():  
                if edit_distance_at_most_one(word.lower(), term.lower()):
                    words[i] = abbreviation + '.'
                    break
    name = ' '.join(words)

    return name

if __name__ == "__main__":
    unittest.main()