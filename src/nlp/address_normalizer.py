"""
Address Normalization and Parsing Module
Pure Python implementation without external C dependencies
Enhanced with fuzzy matching, phonetic algorithms, and Levenshtein distance
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from difflib import SequenceMatcher
from collections import OrderedDict
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comprehensive street type abbreviations
STREET_ABBREVIATIONS = {
    # Primary street types
    'street': 'St',
    'avenue': 'Ave',
    'road': 'Rd',
    'boulevard': 'Blvd',
    'drive': 'Dr',
    'lane': 'Ln',
    'way': 'Way',
    'court': 'Ct',
    'place': 'Pl',
    'circle': 'Cir',
    'trail': 'Trl',
    'parkway': 'Pkwy',
    'highway': 'Hwy',
    'terrace': 'Ter',
    'plaza': 'Plz',
    'square': 'Sq',
    'loop': 'Loop',
    'crescent': 'Cres',
    'alley': 'Aly',
    'path': 'Path',
    'walk': 'Walk',
    'row': 'Row',
    'pike': 'Pike',
    'ridge': 'Rdg',
    'vista': 'Vst',
    'point': 'Pt',
    'crossing': 'Xing',
    'junction': 'Jct',
    'passage': 'Psge',
    'freeway': 'Fwy',
    'expressway': 'Expy',
    'turnpike': 'Tpke',
    'route': 'Rte',
    # Common variations
    'str': 'St',
    'st.': 'St',
    'ave.': 'Ave',
    'av': 'Ave',
    'rd.': 'Rd',
    'blvd.': 'Blvd',
    'dr.': 'Dr',
    'ln.': 'Ln',
    'ct.': 'Ct',
    'pl.': 'Pl',
    'cir.': 'Cir',
    'pkwy.': 'Pkwy',
    'hwy.': 'Hwy'
}

# Directional abbreviations
DIRECTIONAL_ABBREVIATIONS = {
    # Full to abbreviated
    'north': 'N',
    'south': 'S',
    'east': 'E',
    'west': 'W',
    'northeast': 'NE',
    'northwest': 'NW',
    'southeast': 'SE',
    'southwest': 'SW',
    # Common variations
    'n.': 'N',
    's.': 'S',
    'e.': 'E',
    'w.': 'W',
    'ne.': 'NE',
    'nw.': 'NW',
    'se.': 'SE',
    'sw.': 'SW'
}

# Unit/Apartment abbreviations
UNIT_ABBREVIATIONS = {
    'apartment': 'Apt',
    'suite': 'Ste',
    'building': 'Bldg',
    'floor': 'Fl',
    'unit': 'Unit',
    'room': 'Rm',
    'department': 'Dept',
    'office': 'Ofc',
    'penthouse': 'Ph',
    'basement': 'Bsmt',
    # Common variations
    'apt.': 'Apt',
    'ste.': 'Ste',
    'bldg.': 'Bldg',
    'fl.': 'Fl',
    'rm.': 'Rm',
    'dept.': 'Dept',
    'ofc.': 'Ofc'
}

# State abbreviations (US)
STATE_ABBREVIATIONS = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
    'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
    'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
    'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
    'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
    'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
    'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
    'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
    'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
    'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
    'wisconsin': 'WI', 'wyoming': 'WY', 'district of columbia': 'DC',
    'puerto rico': 'PR', 'virgin islands': 'VI', 'guam': 'GU'
}

# Extended abbreviation mapping for better fuzzy matching
ABBREVIATION_EXPANSIONS = {
    # Street types
    'st': ['street', 'saint'],
    'ave': ['avenue'],
    'rd': ['road'],
    'blvd': ['boulevard'],
    'dr': ['drive'],
    'ln': ['lane'],
    'ct': ['court'],
    'pl': ['place'],
    'cir': ['circle'],
    'trl': ['trail'],
    'pkwy': ['parkway'],
    'hwy': ['highway'],
    'ter': ['terrace'],
    'plz': ['plaza'],
    'sq': ['square'],
    # Directionals
    'n': ['north'],
    's': ['south'],
    'e': ['east'],
    'w': ['west'],
    'ne': ['northeast', 'north east'],
    'nw': ['northwest', 'north west'],
    'se': ['southeast', 'south east'],
    'sw': ['southwest', 'south west'],
    # Units
    'apt': ['apartment'],
    'ste': ['suite'],
    'bldg': ['building'],
    'fl': ['floor'],
    'rm': ['room'],
    'dept': ['department'],
    'ofc': ['office']
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        The minimum number of edits needed to transform s1 into s2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def metaphone(word: str, max_length: int = 4) -> str:
    """
    Simplified metaphone algorithm for phonetic matching
    
    Args:
        word: Word to encode
        max_length: Maximum length of the encoded string
    
    Returns:
        Phonetic encoding of the word
    """
    if not word:
        return ""
    
    # Convert to uppercase and remove non-alphabetic characters
    word = ''.join(c for c in word.upper() if c.isalpha())
    
    if not word:
        return ""
    
    # Basic transformations for common phonetic patterns
    transformations = [
        # Silent letters
        (r'^KN', 'N'),
        (r'^GN', 'N'),
        (r'^PN', 'N'),
        (r'^AE', 'E'),
        (r'^WR', 'R'),
        # Common substitutions
        (r'MB$', 'M'),
        (r'CK', 'K'),
        (r'PH', 'F'),
        (r'Q', 'K'),
        (r'V', 'F'),
        (r'Z', 'S'),
        (r'X', 'KS'),
        (r'WH', 'W'),
        # Vowel simplification
        (r'[AEIOU]', 'A'),
        # Double consonants
        (r'(.)\1+', r'\1'),
        # TH sounds
        (r'TH', 'T'),
        # CH sounds
        (r'CH', 'K'),
        # SH sounds
        (r'SH', 'S'),
        # SCH sounds
        (r'SCH', 'SK'),
        # C sounds
        (r'C([EIY])', r'S\1'),
        (r'C', 'K'),
        # G sounds
        (r'G([EIY])', r'J\1'),
        (r'G', 'K'),
        # DG sounds
        (r'DG', 'J'),
        # GH sounds
        (r'GH', ''),
        # H after consonants
        (r'([^AEIOU])H', r'\1'),
        # W after consonants
        (r'([^AEIOU])W', r'\1'),
        # Y sounds
        (r'^Y', 'A'),
        (r'Y', 'A'),
    ]
    
    result = word
    for pattern, replacement in transformations:
        result = re.sub(pattern, replacement, result)
    
    # Remove vowels except at the beginning
    if len(result) > 1:
        result = result[0] + re.sub(r'A', '', result[1:])
    
    # Limit length
    return result[:max_length]


def soundex(word: str) -> str:
    """
    Soundex algorithm for phonetic matching
    
    Args:
        word: Word to encode
    
    Returns:
        Soundex code for the word
    """
    if not word:
        return ""
    
    # Convert to uppercase and keep only letters
    word = ''.join(c for c in word.upper() if c.isalpha())
    
    if not word:
        return ""
    
    # Soundex mappings
    mappings = {
        'BFPV': '1',
        'CGJKQSXZ': '2',
        'DT': '3',
        'L': '4',
        'MN': '5',
        'R': '6',
        'AEIOUHWY': '0'  # Vowels and similar sounds
    }
    
    # Keep first letter
    code = word[0]
    
    # Map remaining letters
    for char in word[1:]:
        for key, value in mappings.items():
            if char in key:
                # Don't add duplicates
                if value != '0' and (not code or code[-1] != value):
                    code += value
                break
    
    # Remove zeros (vowels)
    code = code.replace('0', '')
    
    # Pad with zeros if needed
    code = (code + '000')[:4]
    
    return code


def phonetic_similarity(word1: str, word2: str, algorithm: str = 'both') -> float:
    """
    Calculate phonetic similarity between two words
    
    Args:
        word1: First word
        word2: Second word
        algorithm: 'metaphone', 'soundex', or 'both'
    
    Returns:
        Similarity score between 0 and 1
    """
    if not word1 or not word2:
        return 0.0
    
    scores = []
    
    if algorithm in ['metaphone', 'both']:
        meta1 = metaphone(word1)
        meta2 = metaphone(word2)
        if meta1 == meta2:
            scores.append(1.0)
        else:
            # Calculate similarity based on Levenshtein distance
            distance = levenshtein_distance(meta1, meta2)
            max_len = max(len(meta1), len(meta2))
            if max_len > 0:
                scores.append(1.0 - (distance / max_len))
            else:
                scores.append(0.0)
    
    if algorithm in ['soundex', 'both']:
        sound1 = soundex(word1)
        sound2 = soundex(word2)
        if sound1 == sound2:
            scores.append(1.0)
        else:
            # Calculate partial match
            matching_chars = sum(1 for c1, c2 in zip(sound1, sound2) if c1 == c2)
            scores.append(matching_chars / 4.0)  # Soundex codes are 4 characters
    
    return sum(scores) / len(scores) if scores else 0.0


def expand_abbreviations(text: str) -> List[str]:
    """
    Generate possible expansions of abbreviated text
    
    Args:
        text: Text possibly containing abbreviations
    
    Returns:
        List of possible expanded versions
    """
    if not text:
        return [text]
    
    words = text.lower().split()
    expansions = [[]]
    
    for word in words:
        word_clean = word.strip('.,;')
        new_expansions = []
        
        # Check if word has known expansions
        if word_clean in ABBREVIATION_EXPANSIONS:
            for expansion_list in expansions:
                # Add original
                new_expansions.append(expansion_list + [word])
                # Add expansions
                for expanded in ABBREVIATION_EXPANSIONS[word_clean]:
                    new_expansions.append(expansion_list + [expanded])
        else:
            # No expansion, just add the word
            for expansion_list in expansions:
                new_expansions.append(expansion_list + [word])
        
        expansions = new_expansions
    
    # Convert back to strings and remove duplicates
    result = []
    seen = set()
    for expansion in expansions:
        expanded_text = ' '.join(expansion)
        if expanded_text not in seen:
            result.append(expanded_text)
            seen.add(expanded_text)
    
    return result[:5]  # Limit to 5 most likely expansions


def standardize_abbreviations(text: str) -> str:
    """
    Convert common abbreviations to standard forms.
    
    Args:
        text: Input text with potential abbreviations
        
    Returns:
        Text with standardized abbreviations
    """
    if not text:
        return ""
    
    # Preserve original text for comparison
    result = text
    
    # Split into words for processing
    words = result.split()
    standardized_words = []
    
    for i, word in enumerate(words):
        word_lower = word.lower().strip('.,;')
        replaced = False
        
        # Check for street abbreviations
        if word_lower in STREET_ABBREVIATIONS:
            standardized_words.append(STREET_ABBREVIATIONS[word_lower])
            replaced = True
        # Check for directionals
        elif word_lower in DIRECTIONAL_ABBREVIATIONS:
            standardized_words.append(DIRECTIONAL_ABBREVIATIONS[word_lower])
            replaced = True
        # Check for unit abbreviations
        elif word_lower in UNIT_ABBREVIATIONS:
            standardized_words.append(UNIT_ABBREVIATIONS[word_lower])
            replaced = True
        # Check for state abbreviations (only if it's likely a state context)
        elif i > 0 and word_lower in STATE_ABBREVIATIONS:
            # Check if previous word might be a city name
            prev_word = words[i-1].lower()
            if not prev_word in STREET_ABBREVIATIONS and not prev_word in DIRECTIONAL_ABBREVIATIONS:
                standardized_words.append(STATE_ABBREVIATIONS[word_lower])
                replaced = True
        
        if not replaced:
            # Keep original word with proper capitalization
            if word_lower in ['and', 'of', 'the', 'at', 'in', 'on']:
                standardized_words.append(word.lower())
            else:
                standardized_words.append(word)
    
    return ' '.join(standardized_words)


def normalize_address(address: str) -> str:
    """
    Main normalization function for addresses.
    
    Args:
        address: Raw address string
        
    Returns:
        Normalized address string
    """
    if not address:
        return ""
    
    # Remove extra whitespace and common punctuation
    normalized = re.sub(r'\s+', ' ', address.strip())
    
    # CRITICAL FIX: Remove comma after house number at the beginning of address
    # This fixes addresses like "45, Seapointe Lane" -> "45 Seapointe Lane"
    normalized = re.sub(r'^(\d{1,5}[A-Za-z]?),\s+', r'\1 ', normalized)
    
    # Remove periods after common abbreviations and add space if needed
    # This handles cases like "St.", "Apt.", and "Apt.5" -> "Apt 5"
    for abbrev in ['St', 'Ave', 'Rd', 'Blvd', 'Dr', 'Ln', 'Ct', 'Pl', 'Cir', 'Apt', 'Ste', 'Bldg']:
        # First handle cases where period is followed by non-space (e.g., "Apt.5")
        normalized = re.sub(rf'\b{abbrev}\.([^\s])', rf'{abbrev} \1', normalized, flags=re.IGNORECASE)
        # Then remove remaining periods after abbreviations
        normalized = re.sub(rf'\b{abbrev}\.', abbrev, normalized, flags=re.IGNORECASE)
    
    # Handle comma + period (e.g., "St.,") 
    normalized = re.sub(r'\.(\s*,)', r'\1', normalized)
    
    # Add space after commas if missing
    normalized = re.sub(r',([^\s])', r', \1', normalized)
    
    # Normalize spacing
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Split into parts for processing
    parts = normalized.split(',')
    normalized_parts = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Apply title case selectively
        words = part.split()
        titled_words = []
        for i, word in enumerate(words):
            # Check if it's an all-caps abbreviation (like USA, NYC)
            if word.isupper() and len(word) <= 4:
                titled_words.append(word)
            # Check if it's a number with ordinal suffix
            elif re.match(r'^\d+(st|nd|rd|th)$', word, re.IGNORECASE):
                titled_words.append(word.lower())
            # Check if it's a unit number (e.g., #5, 5A, A5)
            elif re.match(r'^#?\d+[A-Za-z]?$|^[A-Za-z]\d+$', word):
                titled_words.append(word.upper())
            # Apply title case
            else:
                titled_words.append(word.title())
        
        part = ' '.join(titled_words)
        
        # Standardize abbreviations
        part = standardize_abbreviations(part)
        
        normalized_parts.append(part)
    
    # Rejoin with proper comma spacing
    result = ', '.join(normalized_parts)
    
    # Final cleanup
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s*,\s*,\s*', ', ', result)  # Remove double commas
    
    # Handle special cases
    # PO Box normalization
    result = re.sub(r'(?i)p\.?\s*o\.?\s*box', 'PO Box', result)
    # Rural Route normalization
    result = re.sub(r'(?i)r\.?\s*r\.?\s*(\d+)', r'RR \1', result)
    result = re.sub(r'(?i)rural\s+route\s+(\d+)', r'RR \1', result)
    
    return result.strip()


def parse_address_components(address: str) -> Dict[str, Optional[str]]:
    """
    Parse address into structured components.
    
    Args:
        address: Address string to parse
        
    Returns:
        Dictionary with parsed components
    """
    if not address:
        return {
            'house_number': None,
            'street_prefix': None,
            'street_name': None,
            'street_type': None,
            'street_suffix': None,
            'unit': None,
            'city': None,
            'state': None,
            'postal_code': None,
            'country': None,
            'full_address': None
        }
    
    # Normalize the address first
    normalized = normalize_address(address)
    
    components = {
        'house_number': None,
        'street_prefix': None,
        'street_name': None,
        'street_type': None,
        'street_suffix': None,
        'unit': None,
        'city': None,
        'state': None,
        'postal_code': None,
        'country': None,
        'full_address': normalized
    }
    
    # CRITICAL FIX: Check if the entire input is just a standalone number
    # This handles cases like "109" or "123A" that should be house numbers
    standalone_number_pattern = r'^(\d{1,5}[A-Za-z]?)$'
    if re.match(standalone_number_pattern, normalized.strip()):
        # This is just a house number, nothing else
        components['house_number'] = normalized.strip()
        return components
    
    # Handle PO Box addresses
    po_box_match = re.search(r'\bPO\s+Box\s+(\d+)\b', normalized, re.IGNORECASE)
    if po_box_match:
        components['house_number'] = po_box_match.group(0)
        # Remove PO Box from the string for further parsing
        normalized = normalized.replace(po_box_match.group(0), '').strip()
    
    # Split by comma for major components
    parts = [p.strip() for p in normalized.split(',')]
    
    if len(parts) > 0:
        # Parse the street address (first part)
        street_part = parts[0]
        
        # Extract house number (including fractional and alphanumeric)
        # Updated pattern to also work without requiring a space after
        house_number_patterns = [
            r'^(\d+[\-/]?\d*[A-Za-z]?)\s+',  # With space after
            r'^(\d+[\-/]?\d*[A-Za-z]?)$'      # Standalone (no space required)
        ]
        for pattern in house_number_patterns:
            house_match = re.match(pattern, street_part)
            if house_match:
                components['house_number'] = house_match.group(1)
                if house_match.end() < len(street_part):
                    street_part = street_part[house_match.end():].strip()
                else:
                    # The entire street_part was just the house number
                    street_part = ''
                break
        
        if street_part:  # Only continue if there's more to parse
            # Extract directional prefix (N, S, E, W, NE, NW, SE, SW)
            dir_prefix_pattern = r'^(N|S|E|W|NE|NW|SE|SW|North|South|East|West)\s+'
            dir_match = re.match(dir_prefix_pattern, street_part, re.IGNORECASE)
            if dir_match:
                components['street_prefix'] = standardize_abbreviations(dir_match.group(1))
                street_part = street_part[dir_match.end():].strip()
            
            # Extract unit/apartment information
            unit_patterns = [
                r'(?:#|Apt|Apartment|Unit|Ste|Suite)\s*([A-Za-z0-9\-]+)',
                r'(?:Bldg|Building)\s*([A-Za-z0-9\-]+)',
                r'(?:Fl|Floor)\s*([A-Za-z0-9\-]+)'
            ]
            for pattern in unit_patterns:
                unit_match = re.search(pattern, street_part, re.IGNORECASE)
                if unit_match:
                    components['unit'] = unit_match.group(0)
                    street_part = street_part.replace(unit_match.group(0), '').strip()
                    break
            
            # Extract street type and suffix
            street_type_pattern = r'\b(' + '|'.join(re.escape(st) for st in STREET_ABBREVIATIONS.values()) + r')\b'
            street_type_match = re.search(street_type_pattern, street_part, re.IGNORECASE)
            
            if street_type_match:
                components['street_type'] = street_type_match.group(1)
                # Everything before the street type is the street name
                street_name_end = street_type_match.start()
                components['street_name'] = street_part[:street_name_end].strip()
                # Everything after is the suffix (if any)
                suffix_start = street_type_match.end()
                if suffix_start < len(street_part):
                    components['street_suffix'] = street_part[suffix_start:].strip()
            else:
                # No street type found, check if it's just a number (misclassified house number)
                if re.match(r'^\d{1,5}[A-Za-z]?$', street_part.strip()):
                    # This is actually a house number, not a street name
                    if not components['house_number']:
                        components['house_number'] = street_part.strip()
                else:
                    # It's actually a street name
                    components['street_name'] = street_part
    
    if len(parts) > 1:
        # Parse city (usually second part)
        components['city'] = parts[1]
    
    if len(parts) > 2:
        # Parse state and zip (usually third part)
        state_zip_part = parts[2]
        
        # Extract postal code patterns
        postal_patterns = [
            r'\b(\d{5}(?:-\d{4})?)\b',  # US ZIP
            r'\b([A-Z]\d[A-Z]\s?\d[A-Z]\d)\b',  # Canadian
            r'\b([A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2})\b'  # UK
        ]
        
        for pattern in postal_patterns:
            postal_match = re.search(pattern, state_zip_part, re.IGNORECASE)
            if postal_match:
                components['postal_code'] = postal_match.group(1)
                # Remove postal code to isolate state
                state_zip_part = state_zip_part.replace(postal_match.group(0), '').strip()
                break
        
        # What remains should be the state
        if state_zip_part:
            # Check if it's a state abbreviation or full name
            state_normalized = standardize_abbreviations(state_zip_part)
            components['state'] = state_normalized
    
    if len(parts) > 3:
        # Might have country as fourth part
        components['country'] = parts[3]
    
    return components


def fuzzy_match_addresses(addr1: str, addr2: str, 
                         threshold: float = 0.8,
                         component_weights: Optional[Dict[str, float]] = None,
                         use_phonetic: bool = True,
                         use_levenshtein: bool = True) -> Dict[str, Any]:
    """
    Enhanced address matching with fuzzy logic, phonetic algorithms, and Levenshtein distance.
    
    Args:
        addr1: First address
        addr2: Second address  
        threshold: Minimum similarity ratio for component matching (0-1)
        component_weights: Optional custom weights for components
        use_phonetic: Whether to use phonetic matching for street names
        use_levenshtein: Whether to use Levenshtein distance for fuzzy matching
        
    Returns:
        Dictionary with overall score and component-level matching details
    """
    if not addr1 or not addr2:
        return {
            'score': 0.0,
            'confidence': 'none',
            'match_details': {},
            'errors': ['One or both addresses are empty']
        }
    
    # Default component weights with enhanced distribution
    if component_weights is None:
        component_weights = {
            'house_number': 0.30,  # Increased importance
            'street_name': 0.25,
            'street_type': 0.08,
            'city': 0.15,
            'state': 0.10,
            'postal_code': 0.12
        }
    
    # Normalize both addresses
    norm_addr1 = normalize_address(addr1)
    norm_addr2 = normalize_address(addr2)
    
    # Quick check: if normalized addresses are identical
    if norm_addr1.lower() == norm_addr2.lower():
        return {
            'score': 100.0,
            'confidence': 'exact',
            'match_details': {
                'exact_match': True,
                'normalized_addr1': norm_addr1,
                'normalized_addr2': norm_addr2
            },
            'errors': []
        }
    
    # Try abbreviation expansion for better matching
    expansions1 = expand_abbreviations(norm_addr1)
    expansions2 = expand_abbreviations(norm_addr2)
    
    # Check if any expansion combination matches
    for exp1 in expansions1:
        for exp2 in expansions2:
            if exp1.lower() == exp2.lower():
                return {
                    'score': 95.0,
                    'confidence': 'high',
                    'match_details': {
                        'exact_match_after_expansion': True,
                        'expanded_addr1': exp1,
                        'expanded_addr2': exp2
                    },
                    'errors': []
                }
    
    # Parse both addresses
    components1 = parse_address_components(addr1)
    components2 = parse_address_components(addr2)
    
    total_score = 0.0
    total_weight = 0.0
    match_details = {}
    
    for component, weight in component_weights.items():
        val1 = components1.get(component)
        val2 = components2.get(component)
        
        if val1 and val2:
            # Special handling for house numbers
            if component == 'house_number':
                # Exact match gets full score
                if str(val1).lower() == str(val2).lower():
                    component_score = 1.0
                    match_details[component] = 'exact'
                # Check if one contains the other (e.g., "123" vs "123A")
                elif str(val1).lower() in str(val2).lower() or str(val2).lower() in str(val1).lower():
                    component_score = 0.8
                    match_details[component] = 'partial'
                # Check Levenshtein distance for typos
                elif use_levenshtein:
                    distance = levenshtein_distance(str(val1), str(val2))
                    if distance == 1:
                        component_score = 0.6
                        match_details[component] = 'fuzzy_close'
                    elif distance == 2:
                        component_score = 0.3
                        match_details[component] = 'fuzzy_distant'
                    else:
                        component_score = 0.0
                        match_details[component] = 'no_match'
                else:
                    component_score = 0.0
                    match_details[component] = 'no_match'
                    
            # Enhanced street name matching with phonetics
            elif component == 'street_name' and use_phonetic:
                # First try exact match
                if str(val1).lower() == str(val2).lower():
                    component_score = 1.0
                    match_details[component] = 'exact'
                else:
                    # Try phonetic matching
                    phonetic_score = phonetic_similarity(str(val1), str(val2))
                    
                    # Try fuzzy string matching
                    string_ratio = SequenceMatcher(None, 
                                                  str(val1).lower(), 
                                                  str(val2).lower()).ratio()
                    
                    # Use Levenshtein for close matches
                    if use_levenshtein:
                        distance = levenshtein_distance(str(val1).lower(), str(val2).lower())
                        max_len = max(len(str(val1)), len(str(val2)))
                        levenshtein_ratio = 1.0 - (distance / max_len) if max_len > 0 else 0
                    else:
                        levenshtein_ratio = 0
                    
                    # Combine scores with weights
                    component_score = (phonetic_score * 0.4 + 
                                     string_ratio * 0.3 + 
                                     levenshtein_ratio * 0.3)
                    
                    if component_score >= 0.8:
                        match_details[component] = 'strong_fuzzy'
                    elif component_score >= 0.6:
                        match_details[component] = 'moderate_fuzzy'
                    elif component_score >= 0.4:
                        match_details[component] = 'weak_fuzzy'
                    else:
                        match_details[component] = 'no_match'
                        
            # Special handling for postal codes
            elif component == 'postal_code':
                # Compare first 5 digits for US ZIP codes
                zip1 = re.sub(r'[^\d]', '', str(val1))[:5]
                zip2 = re.sub(r'[^\d]', '', str(val2))[:5]
                if zip1 == zip2:
                    component_score = 1.0
                    match_details[component] = 'exact'
                elif zip1[:3] == zip2[:3]:  # Same area code
                    component_score = 0.5
                    match_details[component] = 'area_match'
                else:
                    component_score = 0.0
                    match_details[component] = 'no_match'
                    
            # Fuzzy matching for other components
            else:
                if use_levenshtein:
                    # Use combined approach
                    string_ratio = SequenceMatcher(None, 
                                                  str(val1).lower(), 
                                                  str(val2).lower()).ratio()
                    distance = levenshtein_distance(str(val1).lower(), str(val2).lower())
                    max_len = max(len(str(val1)), len(str(val2)))
                    levenshtein_ratio = 1.0 - (distance / max_len) if max_len > 0 else 0
                    
                    component_score = (string_ratio * 0.6 + levenshtein_ratio * 0.4)
                else:
                    # Just use sequence matcher
                    component_score = SequenceMatcher(None, 
                                                     str(val1).lower(), 
                                                     str(val2).lower()).ratio()
                
                if component_score >= threshold:
                    match_details[component] = 'fuzzy_match'
                else:
                    component_score = 0.0
                    match_details[component] = 'no_match'
            
            total_score += component_score * weight
            total_weight += weight
            match_details[f"{component}_score"] = round(component_score * 100, 1)
            
        elif not val1 and not val2:
            # Both missing - neutral, don't penalize
            match_details[component] = 'both_missing'
        else:
            # One present, one missing - penalize
            total_weight += weight
            match_details[component] = 'one_missing'
            match_details[f"{component}_score"] = 0.0
    
    # Calculate final score
    if total_weight > 0:
        final_score = (total_score / total_weight) * 100
    else:
        # Fall back to advanced string similarity
        if use_levenshtein:
            string_ratio = SequenceMatcher(None, norm_addr1.lower(), norm_addr2.lower()).ratio()
            distance = levenshtein_distance(norm_addr1.lower(), norm_addr2.lower())
            max_len = max(len(norm_addr1), len(norm_addr2))
            levenshtein_ratio = 1.0 - (distance / max_len) if max_len > 0 else 0
            final_score = ((string_ratio * 0.6 + levenshtein_ratio * 0.4) * 100)
        else:
            ratio = SequenceMatcher(None, norm_addr1.lower(), norm_addr2.lower()).ratio()
            final_score = ratio * 100
    
    # Determine confidence level
    if final_score >= 90:
        confidence = 'very_high'
    elif final_score >= 75:
        confidence = 'high'
    elif final_score >= 60:
        confidence = 'moderate'
    elif final_score >= 40:
        confidence = 'low'
    else:
        confidence = 'very_low'
    
    logger.debug(f"Enhanced address match score: {final_score:.1f}% (confidence: {confidence}) for:\n  '{addr1}'\n  '{addr2}'")
    
    return {
        'score': round(final_score, 2),
        'confidence': confidence,
        'match_details': match_details,
        'components1': components1,
        'components2': components2,
        'normalized_addr1': norm_addr1,
        'normalized_addr2': norm_addr2,
        'errors': []
    }


def extract_addresses_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract potential addresses from free-form text.
    
    Args:
        text: Text that may contain addresses
        
    Returns:
        List of dictionaries with extracted addresses and confidence scores
    """
    if not text:
        return []
    
    addresses = []
    
    # US Address pattern - comprehensive but not using VERBOSE flag
    # Pattern for standard US addresses
    us_address_pattern = (
        r'(?:^|\n|\.|,)\s*'  # Start of line or after punctuation
        r'(\d{1,6}[\-/]?\d{0,4}[A-Za-z]?)\s+'  # House number
        r'(?:(N|S|E|W|North|South|East|West)\s+)?'  # Optional prefix direction
        r'([A-Z][A-Za-z0-9\s\-\.]{2,30}?)'  # Street name
        r'\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|Circle|Cir|Trail|Trl|Parkway|Pkwy)\.?\s*'  # Street type
        r'(?:(N|S|E|W|NE|NW|SE|SW))?\s*'  # Optional suffix direction
        r'(?:(?:Apt|Apartment|Unit|Suite|Ste|#)\s*[A-Za-z0-9\-]+)?\s*'  # Optional unit
        r'(?:,\s*([A-Z][A-Za-z\s]{2,30}))?\s*'  # Optional city
        r'(?:,\s*([A-Z]{2}))?\s*'  # Optional state
        r'(?:[\s,]+(\d{5}(?:-\d{4})?))?'  # Optional ZIP
    )
    
    # PO Box pattern
    po_box_pattern = r'(?i)\b(P\.?\s*O\.?\s*Box\s+\d+)(?:\s*,\s*([A-Z][A-Za-z\s]+))?(?:\s*,\s*([A-Z]{2}))?(?:\s+(\d{5}(?:-\d{4})?))?'
    
    # Rural Route pattern
    rural_route_pattern = r'(?i)\b((?:RR|Rural\s+Route)\s+\d+\s+Box\s+\d+)(?:\s*,\s*([A-Z][A-Za-z\s]+))?(?:\s*,\s*([A-Z]{2}))?(?:\s+(\d{5}(?:-\d{4})?))?'
    
    patterns = [
        (us_address_pattern, 'us_address', 0.8),
        (po_box_pattern, 'po_box', 0.9),
        (rural_route_pattern, 'rural_route', 0.9)
    ]
    
    for pattern, pattern_type, base_confidence in patterns:
        # Don't use VERBOSE flag anymore since we concatenated the pattern
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            # Reconstruct address from captured groups
            groups = [g for g in match.groups() if g]
            if len(groups) >= 2:  # Need at least house number and street
                full_address = ' '.join(groups)
                full_address = normalize_address(full_address)
                
                # Check if we already have this address
                if not any(a['address'] == full_address for a in addresses):
                    # Calculate confidence based on completeness
                    confidence = base_confidence
                    if len(groups) >= 4:  # Has city/state/zip
                        confidence = min(confidence + 0.1, 1.0)
                    
                    addresses.append({
                        'address': full_address,
                        'components': parse_address_components(full_address),
                        'pattern_type': pattern_type,
                        'confidence': confidence * 100,
                        'context': text[max(0, match.start()-30):min(len(text), match.end()+30)]
                    })
    
    # Sort by confidence
    addresses.sort(key=lambda x: x['confidence'], reverse=True)
    
    return addresses[:10]  # Return top 10 matches


def validate_address_format(address: str) -> Tuple[bool, List[str]]:
    """
    Validate if an address follows standard formatting rules.
    
    Args:
        address: Address to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not address:
        return False, ['Address is empty']
    
    normalized = normalize_address(address)
    components = parse_address_components(normalized)
    
    # Check for minimum required components
    has_street_address = components['house_number'] and components['street_name']
    has_po_box = 'PO Box' in normalized
    
    if not has_street_address and not has_po_box:
        issues.append('Missing street address or PO Box')
    
    # Check for suspicious patterns
    if re.search(r'\b(test|fake|example|lorem|ipsum)\b', address, re.IGNORECASE):
        issues.append('Contains test/placeholder data')
    
    # Check for excessive special characters
    special_char_count = len(re.findall(r'[^A-Za-z0-9\s,.\-#]', address))
    if special_char_count > 5:
        issues.append(f'Contains excessive special characters ({special_char_count})')
    
    # Check for reasonable length
    if len(address) < 10:
        issues.append('Address too short')
    elif len(address) > 200:
        issues.append('Address too long')
    
    # Check for repeated characters (likely data error)
    if re.search(r'(.)\1{4,}', address):
        issues.append('Contains repeated characters')
    
    # Validate postal code format if present
    if components['postal_code']:
        zip_code = components['postal_code']
        # US ZIP validation
        if not re.match(r'^\d{5}(?:-\d{4})?$', zip_code):
            # Try Canadian
            if not re.match(r'^[A-Z]\d[A-Z]\s?\d[A-Z]\d$', zip_code, re.IGNORECASE):
                # Try UK
                if not re.match(r'^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$', zip_code, re.IGNORECASE):
                    issues.append('Invalid postal code format')
    
    # Validate state abbreviation if present
    if components['state']:
        state = components['state']
        if len(state) == 2:
            # Should be uppercase for US states
            if not state.isupper() or state not in STATE_ABBREVIATIONS.values():
                issues.append(f'Invalid state abbreviation: {state}')
    
    return len(issues) == 0, issues


# Helper function for batch processing
def normalize_address_batch(addresses: List[str]) -> List[str]:
    """
    Normalize a batch of addresses.
    
    Args:
        addresses: List of address strings
        
    Returns:
        List of normalized addresses
    """
    return [normalize_address(addr) for addr in addresses]


# Helper function for finding best match
def find_best_matching_address(target: str, candidates: List[str], min_score: float = 70.0) -> Optional[Tuple[str, float]]:
    """
    Find the best matching address from a list of candidates.
    
    Args:
        target: Target address to match
        candidates: List of candidate addresses
        min_score: Minimum score to consider a match (0-100)
        
    Returns:
        Tuple of (best_match_address, score) or None if no good match
    """
    if not target or not candidates:
        return None
    
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        score = fuzzy_match_addresses(target, candidate)
        if score > best_score and score >= min_score:
            best_score = score
            best_match = candidate
    
    if best_match:
        return (best_match, best_score)
    
    return None