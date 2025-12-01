import json
import logging
import os
from openai import OpenAI
from src.utils.resilience import resilient_openai_call, resilience_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the OpenAI client
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@resilient_openai_call
def analyze_property_description(description):
    """
    Analyzes the property description to extract location hints and important details.
    
    Args:
        description (str): The full property description from Airbnb
        
    Returns:
        dict: Analysis results including:
            - key_location_hints: List of extracted location hints
            - distance_mentions: List of mentions of distances to landmarks
            - landmark_mentions: List of mentioned landmarks
            - neighborhood_insights: Insights about the neighborhood
    """
    try:
        if not description or len(description.strip()) < 10:
            logger.warning("Description is too short to analyze")
            return {
                "key_location_hints": [],
                "distance_mentions": [],
                "landmark_mentions": [],
                "neighborhood_insights": "No description provided."
            }
        
        # Create system prompt for the AI
        system_message = """
        You are an expert real estate analyst specializing in extracting precise location information.
        Analyze the Airbnb property description for any location hints, landmarks, 
        or details that could help verify the exact property location.
        
        Please return your analysis in valid JSON format with the following structure:
        {
            "key_location_hints": ["list", "of", "hints"],
            "distance_mentions": ["mentions", "of", "distances"],
            "landmark_mentions": ["landmarks", "mentioned"],
            "neighborhood_insights": "brief summary of neighborhood"
        }
        
        Focus on extracting specific information about:
        1. Nearby streets, intersections, corners
        2. Distance to landmarks (e.g., "5 min to Main Street")
        3. Proximity to specific shops, restaurants, parks
        4. Named buildings, complexes, or developments
        5. Neighborhood characteristics
        
        Be very precise and extract only factual information. Do not include subjective statements.
        Limit each list to a maximum of 5 high-quality items.
        """
        
        # Call the OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Here is the Airbnb property description to analyze:\n\n{description}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Process the response
        analysis = json.loads(response.choices[0].message.content)
        
        logger.info(f"Successfully analyzed property description, found {len(analysis['key_location_hints'])} location hints")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing property description: {str(e)}")
        return {
            "key_location_hints": [],
            "distance_mentions": [],
            "landmark_mentions": [],
            "neighborhood_insights": f"Analysis failed: {str(e)}"
        }

@resilient_openai_call
def analyze_reviews_for_location(reviews):
    """
    Analyzes guest reviews to extract location-related comments and insights.
    
    Args:
        reviews (list): List of review texts
        
    Returns:
        dict: Analysis including:
            - location_comments: Extracted comments specifically about location
            - location_sentiment: Sentiment about the location (positive/negative/neutral)
            - location_accuracy: Assessment of location accuracy based on reviews
            - key_location_mentions: Specific places mentioned in reviews
    """
    try:
        if not reviews or len(reviews) == 0:
            logger.warning("No reviews provided for analysis")
            return {
                "location_comments": [],
                "location_sentiment": "neutral",
                "location_accuracy": "unknown",
                "key_location_mentions": []
            }
        
        # Join reviews with separator for context
        reviews_text = "\n---\n".join(reviews[:20])  # Limit to 20 reviews for API efficiency
        
        # Create system prompt
        system_message = """
        You are an expert in analyzing guest reviews for vacation rentals.
        Focus only on extracting information related to the property's location.
        
        Please return your analysis in valid JSON format with the following structure:
        {
            "location_comments": ["comment1", "comment2", ...],
            "location_sentiment": "positive/negative/neutral",
            "location_accuracy": "accurate/misleading/unknown",
            "key_location_mentions": ["specific", "places", "mentioned"]
        }
        
        Guidelines:
        1. Extract only comments specifically about the property location
        2. Identify whether guests found the location to match the description
        3. Extract any mentions of specific places, landmarks, or streets
        4. Determine overall sentiment about the location (positive/negative/neutral)
        
        Focus on extracting factual information without speculation.
        Limit location_comments to at most 5 most revealing quotes about the location.
        """
        
        # Call the OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Here are guest reviews to analyze for location information:\n\n{reviews_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Process the response
        analysis = json.loads(response.choices[0].message.content)
        
        logger.info(f"Successfully analyzed reviews, found {len(analysis['location_comments'])} location comments")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing reviews: {str(e)}")
        return {
            "location_comments": [],
            "location_sentiment": "neutral",
            "location_accuracy": "unknown",
            "key_location_mentions": []
        }

@resilient_openai_call
def generate_neighborhood_insights(address, latitude, longitude):
    """
    Generates insights about the neighborhood based on the address and coordinates.
    
    Args:
        address (str): The full property address
        latitude (float): The latitude coordinate
        longitude (float): The longitude coordinate
        
    Returns:
        dict: Neighborhood insights including:
            - neighborhood_overview: General overview of the area
            - points_of_interest: Notable places in the area
            - area_characteristics: Key characteristics of the neighborhood
            - travel_tips: Travel tips for visitors to the area
    """
    try:
        # Create prompt for the AI
        prompt = f"""
        I need insights about the following location:
        
        Address: {address}
        Coordinates: {latitude}, {longitude}
        
        Please provide a comprehensive analysis of this neighborhood based on publicly available information.
        """
        
        system_message = """
        You are a local area expert with deep knowledge of neighborhoods around the world.
        Based on the provided address and coordinates, generate insights about the neighborhood.
        
        Please return your analysis in valid JSON format with the following structure:
        {
            "neighborhood_overview": "brief overview of the area",
            "points_of_interest": ["point1", "point2", "point3"],
            "area_characteristics": ["characteristic1", "characteristic2", "characteristic3"],
            "travel_tips": ["tip1", "tip2", "tip3"]
        }
        
        Guidelines:
        1. Be factual and informative about the general area
        2. Include notable landmarks, attractions, or natural features
        3. Describe the character and style of the neighborhood
        4. Provide practical tips for travelers to this area
        
        Limit each list to 3-5 items maximum. Keep the overview concise (2-3 sentences).
        Only include information that would be publicly known about the general area.
        """
        
        # Call the OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        # Process the response
        insights = json.loads(response.choices[0].message.content)
        
        logger.info(f"Successfully generated neighborhood insights for: {address}")
        return insights
        
    except Exception as e:
        logger.error(f"Error generating neighborhood insights: {str(e)}")
        return {
            "neighborhood_overview": f"Unable to generate neighborhood insights: {str(e)}",
            "points_of_interest": [],
            "area_characteristics": [],
            "travel_tips": []
        }

@resilient_openai_call
def compare_listed_vs_actual_location(listed_location, actual_address):
    """
    Compares the listed location from Airbnb with the actual geocoded address
    to identify any discrepancies or potential misleading information.
    
    Args:
        listed_location (str): The location as described on Airbnb
        actual_address (str): The actual geocoded address
        
    Returns:
        dict: Analysis of the comparison including:
            - match_assessment: How well the locations match (exact/close/different)
            - discrepancies: List of specific discrepancies identified
            - verification_notes: Notes about the verification process
    """
    try:
        if not listed_location or not actual_address:
            logger.warning("Missing listed location or actual address for comparison")
            return {
                "match_assessment": "unknown",
                "discrepancies": [],
                "verification_notes": "Insufficient information for comparison."
            }
        
        # Create prompt for the AI
        prompt = f"""
        I need to compare two locations for an Airbnb property:
        
        LISTED LOCATION (from Airbnb): {listed_location}
        
        ACTUAL ADDRESS (geocoded): {actual_address}
        
        Please analyze if these match and identify any discrepancies.
        """
        
        system_message = """
        You are an expert in geographic verification, specializing in comparing listed locations with actual addresses.
        Your task is to analyze whether an Airbnb listed location matches the actual geocoded address.
        
        Please return your analysis in valid JSON format with the following structure:
        {
            "match_assessment": "exact/close/different",
            "discrepancies": ["discrepancy1", "discrepancy2", ...],
            "verification_notes": "detailed notes on the verification"
        }
        
        Guidelines for match_assessment:
        - "exact" = The locations appear to be identical or virtually identical
        - "close" = The locations are in the same general area but have minor differences
        - "different" = The locations have significant differences that could affect a guest's stay
        
        For discrepancies, focus on differences in:
        1. Neighborhood or district names
        2. Proximity to key landmarks or attractions
        3. Geographic features (beach access, mountain views, etc.)
        4. Urban vs. suburban vs. rural character
        
        Be precise and factual. Do not speculate beyond the information provided.
        """
        
        # Call the OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Process the response
        comparison = json.loads(response.choices[0].message.content)
        
        logger.info(f"Successfully compared locations: match assessment = {comparison['match_assessment']}")
        return comparison
        
    except Exception as e:
        logger.error(f"Error comparing locations: {str(e)}")
        return {
            "match_assessment": "error",
            "discrepancies": [],
            "verification_notes": f"Error during comparison: {str(e)}"
        }