"""
NLP Extractor Module for Airbnb Listing Descriptions
Extracts street names, HOA names, and POIs from property descriptions
using regex patterns and confidence scoring.
"""

import re
import logging
from typing import List, Dict, Tuple, Any, Optional
from nlp.address_normalizer import normalize_address, standardize_abbreviations

logger = logging.getLogger(__name__)

# Street suffixes and their variations
STREET_SUFFIXES = {
    'Street': ['street', 'st', 'str'],
    'Avenue': ['avenue', 'ave', 'av'],
    'Road': ['road', 'rd'],
    'Boulevard': ['boulevard', 'blvd', 'boul', 'bvd'],
    'Drive': ['drive', 'dr', 'drv'],
    'Lane': ['lane', 'ln'],
    'Way': ['way', 'wy'],
    'Court': ['court', 'ct'],
    'Place': ['place', 'pl', 'plc'],
    'Circle': ['circle', 'cir', 'crcl'],
    'Trail': ['trail', 'trl', 'tr'],
    'Parkway': ['parkway', 'pkwy', 'pky'],
    'Highway': ['highway', 'hwy'],
    'Terrace': ['terrace', 'ter', 'terr'],
    'Plaza': ['plaza', 'plz'],
    'Square': ['square', 'sq'],
    'Loop': ['loop', 'lp'],
    'Crescent': ['crescent', 'cres', 'cr'],
    'Alley': ['alley', 'aly'],
    'Path': ['path'],
    'Walk': ['walk'],
    'Row': ['row'],
    'Pike': ['pike'],
    'Ridge': ['ridge', 'rdg'],
    'Vista': ['vista', 'vst']
}

# Common HOA/Community keywords
HOA_KEYWORDS = [
    'community', 'neighborhood', 'hoa', 'homeowners association',
    'association', 'subdivision', 'estates', 'estate', 'village',
    'complex', 'development', 'residences', 'residence', 'commons',
    'park', 'hills', 'heights', 'gardens', 'grove', 'manor',
    'meadows', 'pointe', 'preserve', 'ranch', 'reserve', 'ridge',
    'springs', 'valley', 'villas', 'villa', 'woods', 'acres',
    'club', 'cove', 'crossing', 'enclave', 'greens', 'haven',
    'hollow', 'knoll', 'landing', 'oaks', 'palms', 'pines',
    'plantation', 'shores', 'terrace', 'towers', 'trace', 'trails',
    'vineyards', 'waters', 'bay', 'beach', 'bluff', 'island'
]

# Common POI types
POI_TYPES = {
    'parks': ['park', 'recreation area', 'green space', 'playground', 'nature reserve'],
    'beaches': ['beach', 'shore', 'oceanfront', 'waterfront', 'coastline', 'seaside', 'lakefront'],
    'airports': ['airport', 'international airport', 'regional airport', 'airfield'],
    'universities': ['university', 'college', 'campus', 'institute', 'academy'],
    'hospitals': ['hospital', 'medical center', 'clinic', 'healthcare', 'emergency room'],
    'shopping': ['shopping center', 'mall', 'shopping mall', 'outlet', 'marketplace', 
                 'shopping district', 'retail', 'stores', 'shops', 'boutiques'],
    'restaurants': ['restaurant', 'cafe', 'diner', 'bistro', 'eatery', 'dining', 
                   'food court', 'bar', 'pub', 'grill'],
    'transportation': ['station', 'metro', 'subway', 'train station', 'bus stop', 
                      'transit', 'terminal', 'ferry'],
    'entertainment': ['theater', 'theatre', 'cinema', 'museum', 'gallery', 'arena', 
                     'stadium', 'convention center', 'casino', 'amusement park'],
    'landmarks': ['landmark', 'monument', 'memorial', 'historic site', 'tourist attraction',
                 'point of interest', 'bridge', 'tower', 'plaza', 'square'],
    'schools': ['school', 'elementary', 'middle school', 'high school', 'primary school'],
    'religious': ['church', 'cathedral', 'temple', 'mosque', 'synagogue', 'chapel'],
    'government': ['city hall', 'courthouse', 'post office', 'library', 'civic center'],
    'sports': ['golf course', 'tennis court', 'gym', 'fitness center', 'sports complex',
              'country club', 'marina', 'yacht club']
}

def calculate_confidence(match_type: str, pattern_type: str, context_words: List[str]) -> float:
    """
    Calculate confidence score based on match type and context.
    
    Args:
        match_type: Type of match ('exact', 'partial', 'fuzzy')
        pattern_type: Type of pattern ('street', 'hoa', 'poi')
        context_words: Surrounding context words
        
    Returns:
        Confidence score between 0 and 100
    """
    base_scores = {
        'exact': 90,
        'partial': 70,
        'fuzzy': 50
    }
    
    score = base_scores.get(match_type, 50)
    
    # Boost score for strong context indicators
    strong_indicators = {
        'street': ['located on', 'situated on', 'address', 'corner of', 'intersection'],
        'hoa': ['gated', 'secured', 'exclusive', 'private', 'managed by'],
        'poi': ['walking distance', 'minutes from', 'blocks from', 'next to', 'across from']
    }
    
    context_lower = ' '.join(context_words).lower()
    for indicator in strong_indicators.get(pattern_type, []):
        if indicator in context_lower:
            score = min(score + 10, 100)
            break
    
    # Reduce score for weak or negative indicators
    weak_indicators = ['like', 'similar to', 'reminds of', 'feels like', 'style']
    for indicator in weak_indicators:
        if indicator in context_lower:
            score = max(score - 20, 30)
            break
    
    return score

def extract_street_names(text: str) -> List[Dict[str, Any]]:
    """
    Enhanced street name extraction with international support and context awareness.
    
    Args:
        text: Property description text
        
    Returns:
        List of dictionaries with 'street_name', 'confidence', 'context', and 'address_parts'
    """
    if not text:
        return []
    
    results = []
    seen = set()
    
    # Build regex pattern for street suffixes
    suffix_patterns = []
    for standard, variations in STREET_SUFFIXES.items():
        suffix_patterns.extend(variations)
    suffix_regex = r'\b(' + '|'.join(re.escape(s) for s in suffix_patterns) + r')\b'
    
    # Pattern 1: Full address with house number, street, and optional unit
    pattern_full_address = rf'(\d{{1,5}}[A-Za-z]?)\s+([A-Z][A-Za-z\s]*?)\s+{suffix_regex}(?:\s*,?\s*(?:Apt|Unit|Suite|#)\s*[A-Za-z0-9\-]+)?'
    
    # Pattern 2: "on/at/near [Street Name] [Suffix]" with context
    pattern_context = rf'(?:located\s+)?(?:on|at|near|off|along|corner\s+of)\s+([A-Z][A-Za-z\s]*?)\s+{suffix_regex}'
    
    # Pattern 3: "[Street Name] [Suffix]" with proper capitalization
    pattern_standard = rf'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*?)\s+{suffix_regex}'
    
    # Pattern 4: Numbered streets (e.g., "5th Avenue", "42nd Street", "1st Street")
    pattern_numbered = rf'\b(\d{{1,3}}(?:st|nd|rd|th))\s+{suffix_regex}'
    
    # Pattern 5: "[Direction] [Street Name] [Suffix]" (e.g., "North Main Street")
    pattern_directional = rf'\b((?:North|South|East|West|N\.?|S\.?|E\.?|W\.?|NE|NW|SE|SW)\s+[A-Z][A-Za-z\s]*?)\s+{suffix_regex}'
    
    # Pattern 6: Streets with "and" or "&" (intersections)
    pattern_intersection = rf'([A-Z][A-Za-z\s]*?)\s+{suffix_regex}\s+(?:and|&|\@)\s+([A-Z][A-Za-z\s]*?)\s+{suffix_regex}'
    
    # Pattern 7: International formats (e.g., "Rue de [Name]", "Via [Name]")
    pattern_international = r'(?:Rue|Via|Calle|Avenida|Strasse|Straße)\s+(?:de\s+|del\s+|des\s+)?([A-Z][A-Za-z\s]+?)(?:\.|,|;|\s+(?:and|with|near))'
    
    # Pattern 8: Highway/Route patterns (e.g., "Highway 101", "Route 66", "I-95")
    pattern_highway = r'(?:Highway|Route|Interstate|I-|US-|State Route|SR-|FM-)\s*(\d+[A-Za-z]?)'
    
    # Pattern 9: Between two streets pattern
    pattern_between = rf'between\s+([A-Z][A-Za-z\s]*?)\s+{suffix_regex}\s+and\s+([A-Z][A-Za-z\s]*?)\s+{suffix_regex}'
    
    # Pattern 10: Address with apartment/unit inline
    pattern_with_unit = rf'(\d{{1,5}}[A-Za-z]?)\s+([A-Z][A-Za-z\s]*?)\s+{suffix_regex}(?:\s*(?:Apt|Unit|Suite|#)\s*([A-Za-z0-9\-]+))?'
    
    patterns = [
        (pattern_full_address, 'exact', 1, 'full_address'),
        (pattern_context, 'exact', 1, 'context'),
        (pattern_standard, 'partial', 1, 'standard'),
        (pattern_numbered, 'exact', 1, 'numbered'),
        (pattern_directional, 'exact', 1, 'directional'),
        (pattern_intersection, 'exact', 2, 'intersection'),
        (pattern_international, 'partial', 1, 'international'),
        (pattern_highway, 'exact', 1, 'highway'),
        (pattern_between, 'exact', 2, 'between'),
        (pattern_with_unit, 'exact', 1, 'with_unit')
    ]
    
    for pattern, match_type, capture_count, pattern_name in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            address_parts = {}
            
            if pattern_name == 'full_address':
                # Handle full address pattern
                house_number = match.group(1)
                street_name = f"{match.group(2)} {match.group(3)}".strip()
                address_parts = {'house_number': house_number, 'street_name': street_name}
                if street_name.lower() not in seen and len(street_name) > 3:
                    seen.add(street_name.lower())
                    start = max(0, match.start() - 60)
                    end = min(len(text), match.end() + 60)
                    context = text[start:end]
                    confidence = calculate_confidence(match_type, 'street', context.split())
                    confidence = min(confidence + 10, 100)  # Boost for full address
                    normalized_street = standardize_abbreviations(street_name.title())
                    results.append({
                        'street_name': normalized_street,
                        'house_number': house_number,
                        'confidence': confidence,
                        'context': context.strip(),
                        'pattern': pattern_name,
                        'address_parts': address_parts
                    })
                    
            elif pattern_name == 'with_unit':
                # Handle address with unit pattern
                house_number = match.group(1)
                street_name = f"{match.group(2)} {match.group(3)}".strip()
                unit = match.group(4) if len(match.groups()) >= 4 else None
                address_parts = {'house_number': house_number, 'street_name': street_name, 'unit': unit}
                if street_name.lower() not in seen and len(street_name) > 3:
                    seen.add(street_name.lower())
                    start = max(0, match.start() - 60)
                    end = min(len(text), match.end() + 60)
                    context = text[start:end]
                    confidence = calculate_confidence(match_type, 'street', context.split())
                    confidence = min(confidence + 10, 100)  # Boost for complete address
                    normalized_street = standardize_abbreviations(street_name.title())
                    results.append({
                        'street_name': normalized_street,
                        'house_number': house_number,
                        'unit': unit,
                        'confidence': confidence,
                        'context': context.strip(),
                        'pattern': pattern_name,
                        'address_parts': address_parts
                    })
                    
            elif pattern_name in ['intersection', 'between']:
                # Handle two-street patterns
                street1 = f"{match.group(1)} {match.group(2)}"
                street2 = f"{match.group(3)} {match.group(4)}"
                for street in [street1, street2]:
                    street = street.strip()
                    if street.lower() not in seen and len(street) > 3:
                        seen.add(street.lower())
                        start = max(0, match.start() - 60)
                        end = min(len(text), match.end() + 60)
                        context = text[start:end]
                        confidence = calculate_confidence(match_type, 'street', context.split())
                        normalized_street = standardize_abbreviations(street.title())
                        results.append({
                            'street_name': normalized_street,
                            'confidence': confidence,
                            'context': context.strip(),
                            'pattern': pattern_name,
                            'address_parts': {'intersection': True}
                        })
                        
            elif pattern_name == 'highway':
                # Handle highway pattern
                highway_name = f"Highway {match.group(1)}" if "Highway" in pattern else f"Route {match.group(1)}"
                if highway_name.lower() not in seen:
                    seen.add(highway_name.lower())
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    confidence = calculate_confidence(match_type, 'street', context.split())
                    results.append({
                        'street_name': highway_name,
                        'confidence': confidence,
                        'context': context.strip(),
                        'pattern': pattern_name,
                        'address_parts': {'highway': True}
                    })
                    
            elif pattern_name == 'international':
                # Handle international format
                street_type = match.group(0).split()[0]  # Rue, Via, etc.
                street_name = f"{street_type} {match.group(1)}".strip()
                if street_name.lower() not in seen and len(street_name) > 3:
                    seen.add(street_name.lower())
                    start = max(0, match.start() - 60)
                    end = min(len(text), match.end() + 60)
                    context = text[start:end]
                    confidence = calculate_confidence(match_type, 'street', context.split())
                    results.append({
                        'street_name': street_name.title(),
                        'confidence': confidence,
                        'context': context.strip(),
                        'pattern': pattern_name,
                        'address_parts': {'international': True}
                    })
                    
            else:
                # Handle standard patterns
                if len(match.groups()) >= 2:
                    street = f"{match.group(1)} {match.group(2)}".strip()
                else:
                    street = match.group(1).strip() if match.group(1) else ""
                    
                if street and street.lower() not in seen and len(street) > 3:
                    seen.add(street.lower())
                    start = max(0, match.start() - 60)
                    end = min(len(text), match.end() + 60)
                    context = text[start:end]
                    confidence = calculate_confidence(match_type, 'street', context.split())
                    
                    # Boost confidence for context patterns
                    if pattern_name == 'context':
                        confidence = min(confidence + 5, 100)
                    elif pattern_name == 'directional':
                        confidence = min(confidence + 3, 100)
                        
                    normalized_street = standardize_abbreviations(street.title())
                    results.append({
                        'street_name': normalized_street,
                        'confidence': confidence,
                        'context': context.strip(),
                        'pattern': pattern_name,
                        'address_parts': {}
                    })
    
    # Sort by confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results[:10]  # Return top 10 matches

def extract_hoa_names(text: str) -> List[Dict[str, Any]]:
    """
    Extract HOA and community names from property description text.
    
    Args:
        text: Property description text
        
    Returns:
        List of dictionaries with 'hoa_name', 'confidence', and 'context'
    """
    if not text:
        return []
    
    results = []
    seen = set()
    
    # Build regex pattern for HOA keywords
    hoa_pattern = r'\b(' + '|'.join(re.escape(k) for k in HOA_KEYWORDS) + r')\b'
    
    # Pattern 1: "in [Name] [HOA Keyword]" (e.g., "in Sunset Hills Community")
    pattern1 = rf'(?:in|at|within)\s+(?:the\s+)?([A-Z][A-Za-z\s&]+?)\s+{hoa_pattern}'
    
    # Pattern 2: "[Name] [HOA Keyword]" with proper capitalization
    pattern2 = rf'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*?)\s+{hoa_pattern}'
    
    # Pattern 3: "The [Name] [HOA Keyword]"
    pattern3 = rf'\bThe\s+([A-Z][A-Za-z\s&]+?)\s+{hoa_pattern}'
    
    # Pattern 4: Gated/Private community patterns
    pattern4 = r'(?:gated|private|exclusive|secured?)\s+(?:community|neighborhood|development)\s+(?:of\s+)?([A-Z][A-Za-z\s&]+?)(?:\.|,|;|\s+(?:with|featuring|offering))'
    
    # Pattern 5: "[Name]" (in quotes, often indicates proper names)
    pattern5 = r'"([A-Z][A-Za-z\s&]+?)"(?:\s+(?:community|neighborhood|development|estates?|village))?'
    
    patterns = [
        (pattern1, 'exact'),
        (pattern2, 'partial'),
        (pattern3, 'exact'),
        (pattern4, 'exact'),
        (pattern5, 'partial')
    ]
    
    for pattern, match_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            # Get the HOA name from the appropriate group
            if len(match.groups()) >= 2:
                hoa_name = match.group(1).strip()
                keyword = match.group(2) if len(match.groups()) >= 2 else ''
            else:
                hoa_name = match.group(1).strip()
                keyword = ''
            
            # Clean up the name
            hoa_name = re.sub(r'\s+', ' ', hoa_name)  # Normalize whitespace
            
            if hoa_name.lower() not in seen and len(hoa_name) > 2:
                seen.add(hoa_name.lower())
                
                # Get context
                start = max(0, match.start() - 60)
                end = min(len(text), match.end() + 60)
                context = text[start:end]
                
                # Calculate confidence
                confidence = calculate_confidence(match_type, 'hoa', context.split())
                
                # Boost confidence if it contains multiple HOA keywords
                keyword_count = sum(1 for k in HOA_KEYWORDS if k in hoa_name.lower())
                if keyword_count > 1:
                    confidence = min(confidence + 10, 100)
                
                results.append({
                    'hoa_name': hoa_name.title(),
                    'type': keyword.title() if keyword else 'Community',
                    'confidence': confidence,
                    'context': context.strip()
                })
    
    # Sort by confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results[:10]  # Return top 10 matches

def extract_pois(text: str) -> List[Dict[str, Any]]:
    """
    Extract Points of Interest (POIs) from property description text.
    
    Args:
        text: Property description text
        
    Returns:
        List of dictionaries with 'poi_name', 'type', 'distance', 'confidence', and 'context'
    """
    if not text:
        return []
    
    results = []
    seen = set()
    
    # Build POI type patterns
    poi_patterns = []
    for poi_type, keywords in POI_TYPES.items():
        for keyword in keywords:
            poi_patterns.append((keyword, poi_type))
    
    # Distance patterns
    distance_pattern = r'(\d+(?:\.\d+)?)\s*(?:min(?:ute)?s?|mi(?:le)?s?|km|meters?|blocks?|feet|ft|minutes?\s+walk|minutes?\s+drive)'
    
    # Pattern 1: "[Distance] from/to [POI Name]"
    pattern1 = rf'{distance_pattern}\s+(?:from|to|away from)\s+(?:the\s+)?([A-Z][A-Za-z\s&\']+?)(?:\.|,|;|\s+(?:and|with|where))'
    
    # Pattern 2: "near/close to [POI Name]"
    pattern2 = r'(?:near|close to|adjacent to|next to|across from|behind|in front of)\s+(?:the\s+)?([A-Z][A-Za-z\s&\']+?)(?:\.|,|;|\s+(?:and|with|where))'
    
    # Pattern 3: "walk/drive to [POI Name]"
    pattern3 = r'(?:walk|drive|stroll|bike)\s+to\s+(?:the\s+)?([A-Z][A-Za-z\s&\']+?)(?:\.|,|;|\s+(?:and|with|in))'
    
    # Pattern 4: Specific POI type mentions with names
    poi_type_patterns = []
    for poi_type, keywords in POI_TYPES.items():
        keyword_pattern = '|'.join(re.escape(k) for k in keywords)
        poi_type_patterns.append(
            (rf'([A-Z][A-Za-z\s&\']+?)\s+({keyword_pattern})', poi_type)
        )
    
    # Process distance-based patterns
    for pattern in [pattern1]:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            distance = match.group(1)
            poi_name = match.group(2).strip()
            
            if poi_name.lower() not in seen and len(poi_name) > 2:
                seen.add(poi_name.lower())
                
                # Determine POI type
                poi_type = 'general'
                for keywords, ptype in poi_patterns:
                    if keywords.lower() in poi_name.lower():
                        poi_type = ptype
                        break
                
                # Get context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                confidence = calculate_confidence('exact', 'poi', context.split())
                
                results.append({
                    'poi_name': poi_name.title(),
                    'type': poi_type,
                    'distance': distance,
                    'confidence': confidence,
                    'context': context.strip()
                })
    
    # Process proximity patterns (no specific distance)
    for pattern in [pattern2, pattern3]:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            poi_name = match.group(1).strip()
            
            if poi_name.lower() not in seen and len(poi_name) > 2:
                seen.add(poi_name.lower())
                
                # Determine POI type
                poi_type = 'general'
                for keywords, ptype in poi_patterns:
                    if keywords.lower() in poi_name.lower():
                        poi_type = ptype
                        break
                
                # Get context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                confidence = calculate_confidence('partial', 'poi', context.split())
                
                results.append({
                    'poi_name': poi_name.title(),
                    'type': poi_type,
                    'distance': 'nearby',
                    'confidence': confidence,
                    'context': context.strip()
                })
    
    # Process POI type patterns
    for pattern_template, poi_type in poi_type_patterns:
        matches = re.finditer(pattern_template, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            poi_name = f"{match.group(1)} {match.group(2)}".strip()
            
            if poi_name.lower() not in seen and len(poi_name) > 3:
                seen.add(poi_name.lower())
                
                # Get context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Look for distance information nearby
                distance = 'unknown'
                distance_match = re.search(distance_pattern, context, re.IGNORECASE)
                if distance_match:
                    distance = distance_match.group(0)
                
                confidence = calculate_confidence('exact', 'poi', context.split())
                
                results.append({
                    'poi_name': poi_name.title(),
                    'type': poi_type,
                    'distance': distance,
                    'confidence': confidence,
                    'context': context.strip()
                })
    
    # Sort by confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results[:15]  # Return top 15 POIs

def extract_nlp_location_data(text: str) -> Dict[str, Any]:
    """
    Extract all location-related entities from property description.
    
    Args:
        text: Property description text
        
    Returns:
        Dictionary containing all extracted location data with confidence scores
    """
    if not text:
        logger.warning("No text provided for NLP extraction")
        return {
            'street_names': [],
            'hoa_names': [],
            'pois': [],
            'overall_confidence': 0,
            'text_length': 0,
            'extraction_summary': {
                'total_streets': 0,
                'total_hoas': 0,
                'total_pois': 0,
                'high_confidence_items': 0
            }
        }
    
    logger.info(f"Extracting NLP location data from text of length {len(text)}")
    
    # Extract entities
    street_names = extract_street_names(text)
    hoa_names = extract_hoa_names(text)
    pois = extract_pois(text)
    
    # Calculate overall confidence based on extracted entities
    confidence_scores = []
    if street_names:
        confidence_scores.extend([s['confidence'] for s in street_names[:3]])
    if hoa_names:
        confidence_scores.extend([h['confidence'] for h in hoa_names[:2]])
    if pois:
        confidence_scores.extend([p['confidence'] for p in pois[:3]])
    
    overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    # Log extraction results
    logger.debug(f"Extracted {len(street_names)} street names, {len(hoa_names)} HOA names, {len(pois)} POIs")
    
    return {
        'street_names': street_names,
        'hoa_names': hoa_names,
        'pois': pois,
        'overall_confidence': round(overall_confidence, 2),
        'text_length': len(text),
        'extraction_summary': {
            'total_streets': len(street_names),
            'total_hoas': len(hoa_names),
            'total_pois': len(pois),
            'high_confidence_items': sum(1 for item in street_names + hoa_names + pois if item.get('confidence', 0) > 80)
        }
    }

def get_best_address_from_nlp(nlp_data: Dict[str, Any], current_address: Optional[str] = None) -> Optional[str]:
    """
    Generate the best address from NLP-extracted data.
    
    Args:
        nlp_data: Dictionary containing NLP extraction results
        current_address: Current address if available
        
    Returns:
        Best address string or None if insufficient data
    """
    if not nlp_data:
        return current_address
    
    # Get the highest confidence street name
    street_names = nlp_data.get('street_names', [])
    best_street = None
    if street_names and street_names[0].get('confidence', 0) > 70:
        best_street = street_names[0].get('street_name')
    
    # Get the highest confidence HOA/community
    hoa_names = nlp_data.get('hoa_names', [])
    best_hoa = None
    if hoa_names and hoa_names[0].get('confidence', 0) > 70:
        best_hoa = hoa_names[0].get('hoa_name')
    
    # Build an enhanced address
    address_parts = []
    
    if best_street:
        address_parts.append(best_street)
    
    if best_hoa and best_hoa not in (best_street or ''):
        address_parts.append(best_hoa)
    
    if current_address:
        # Parse the current address to get city, state, etc.
        current_parts = current_address.split(',')
        if len(current_parts) > 1:
            # Add city, state, country from current address
            address_parts.extend([p.strip() for p in current_parts[-2:]])
    
    if address_parts:
        return ', '.join(address_parts)
    
    return current_address