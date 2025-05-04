"""
Functions for normalizing the names of the interest groups that appear
in the Amici database. 
"""
import re
from typing import List, Optional
import editdistance
import unittest
from hypothesis import given, strategies as st
from hypothesis.strategies import composite
from hypothesis import settings
import numpy as np

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
    'division': 'div',
    'group': 'grp',
    'center': 'ctr'
}

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
    name = re.sub(r'\s+(?:inc|llc|corp|ltd|plc|gmbh|ag|sa)(?:\.|\b)', '', name, flags=re.IGNORECASE)

    # Remove acronyms in parentheses appearing at the end of the name
    name = re.sub(r'\s*\([A-Z]{2,}\)\s*$', '', name)

    # Replace ampersands with 'and'
    name = re.sub(r'&', 'and', name)

    # Convert to lowercase
    name = name.lower()

    # Remove apostrophes
    name = re.sub(r"'", '', name)

    # Remove commas
    name = re.sub(r',', '', name)

    # Shorten common terms
    name = shorten_common_terms(name)

    # Replace all types of dashes with one dash and condense consecutive dashes
    name = re.sub(r'\s*[-–—]+\s*', '-', name)

    name = re.sub(r'\s+', ' ', name).strip()

    return name

def shorten_common_terms(name: str) -> str:
    """
    Shorten common terms in the name of an interest group.

    Args:
        name (str): The name of the interest group.

    Returns:
        str: The shortened name.
    """

    # Shorten common terms
    for term, abbreviation in common_terms.items():
        name = re.sub(r'\b' + term + r'\b', abbreviation, name, flags=re.IGNORECASE)

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
                    words[i] = abbreviation
                    break
    name = ' '.join(words)

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
        whitelist_characters=" ,.-—–()'" + ''.join(['0123456789'])
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



    @composite
    def terms_with_alterations(draw):
        """Strategy to produce terms from common_terms with occasional alterations."""
        terms = draw(st.lists(
            st.sampled_from(list(common_terms.keys())), 
            min_size=0, max_size=5
        ))
        
        altered_terms = []
        for term in terms:
            # 30% chance to alter the term
            if draw(st.sampled_from([True, False, False, False, False, False, False, True, False, True])):
                # Possible alterations: add/remove/change a character
                alteration_type = draw(st.sampled_from(['add', 'remove', 'change']))
                if alteration_type == 'add' and len(term) > 0:
                    pos = draw(st.integers(min_value=0, max_value=len(term)))
                    char = draw(st.characters(whitelist_categories=('Lu', 'Ll')))
                    term = term[:pos] + char + term[pos:]
                elif alteration_type == 'remove' and len(term) > 1:
                    pos = draw(st.integers(min_value=0, max_value=len(term)-1))
                    term = term[:pos] + term[pos+1:]
                elif alteration_type == 'change' and len(term) > 0:
                    pos = draw(st.integers(min_value=0, max_value=len(term)-1))
                    char = draw(st.characters(whitelist_categories=('Lu', 'Ll')))
                    term = term[:pos] + char + term[pos+1:]
            
            altered_terms.append(term)
        
        return altered_terms

    @given(terms_with_alterations())
    def test_common_term_normalization(self, terms):
        """Test that terms from common_terms (possibly altered) get normalized correctly."""
        if not terms:  # Skip empty lists
            return
            
        # Construct a test string with the terms
        test_input = " ".join(terms)
        
        # Normalize the input
        normalized = normalize_interest_group_name(test_input)
        
        # Check that normalizing again produces the same result
        self.assertEqual(normalized, normalize_interest_group_name(normalized))

if __name__ == "__main__":
    
    n1 = normalize_interest_group_name("American Trucking Association's")
    n2 = normalize_interest_group_name(n1)
    print(f"Normalized name: {n1}")
    print(f"Normalized name again: {n2}")

    unittest.main()