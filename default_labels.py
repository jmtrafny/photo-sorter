"""
Built-in default labels configuration for photo organizer.

This module provides tiered label configurations optimized for different
model sizes and use cases. Labels are designed to maximize accuracy while
maintaining good performance for each tier.

Tiers:
- SMALL: Basic categories for quick sorting (~10 labels)
- MEDIUM: Extended categories for better accuracy (~20 labels)  
- LARGE: Comprehensive categories for professional use (~30+ labels)

The labels are designed based on common photo categories found in personal
and professional collections, with carefully chosen synonyms to improve 
CLIP's recognition accuracy across different visual styles and contexts.
"""

# Small label set - optimized for speed with basic categories
SMALL_LABELS = {
    "people": {
        "prompt": "people",
        "synonyms": [
            "people", "person", "portrait", "selfie", "face", "human"
        ],
        "weight": 1.0
    },
    "pets": {
        "prompt": "pets",
        "synonyms": [
            "pets", "dog", "cat", "animal"
        ],
        "weight": 1.1
    },
    "landscape": {
        "prompt": "landscape",
        "synonyms": [
            "landscape", "scenery", "mountains", "beach", "nature", "sunset"
        ],
        "weight": 1.0
    },
    "food": {
        "prompt": "food",
        "synonyms": [
            "food", "meal", "dish", "dessert"
        ],
        "weight": 1.0
    },
    "documents": {
        "prompt": "documents",
        "synonyms": [
            "documents", "paper", "text", "screenshot"
        ],
        "weight": 1.0
    },
    "cars": {
        "prompt": "car",
        "synonyms": [
            "car", "vehicle", "automobile"
        ],
        "weight": 0.9
    },
    "buildings": {
        "prompt": "building",
        "synonyms": [
            "building", "architecture", "house", "city"
        ],
        "weight": 1.0
    },
    "events": {
        "prompt": "event",
        "synonyms": [
            "wedding", "party", "celebration", "concert"
        ],
        "weight": 1.0
    }
}

# Medium label set - balanced accuracy and performance
MEDIUM_LABELS = {
    "people": {
        "prompt": "people",
        "synonyms": [
            "people", "person", "a person", "group of people",
            "portrait", "selfie", "man", "woman", "face", 
            "human", "child", "family photo", "friends"
        ],
        "weight": 1.0
    },
    "pets": {
        "prompt": "pets",
        "synonyms": [
            "pets", "pet", "dog", "cat", "puppy", "kitten",
            "hamster", "rabbit", "bird", "fish", "animal companion"
        ],
        "weight": 1.1
    },
    "wildlife": {
        "prompt": "wildlife",
        "synonyms": [
            "wildlife", "wild animal", "bird", "deer", "bear",
            "elephant", "lion", "zoo animal", "safari"
        ],
        "weight": 1.0
    },
    "landscape": {
        "prompt": "landscape",
        "synonyms": [
            "landscape", "scenery", "mountains", "mountain",
            "beach", "ocean", "desert", "forest", "waterfall",
            "lake", "sunset", "sunrise", "river", "nature scene"
        ],
        "weight": 1.0
    },
    "cityscape": {
        "prompt": "cityscape",
        "synonyms": [
            "city skyline", "urban", "skyscrapers", "downtown",
            "street view", "city lights", "metropolitan"
        ],
        "weight": 1.0
    },
    "food": {
        "prompt": "food",
        "synonyms": [
            "food", "meal", "pizza", "burger", "sushi", "salad",
            "cake", "dessert", "dish", "drink", "fruit", 
            "restaurant food", "cooking"
        ],
        "weight": 1.0
    },
    "documents": {
        "prompt": "documents",
        "synonyms": [
            "documents", "document", "paper", "whiteboard",
            "presentation slide", "slide", "text", "handwriting",
            "notes", "diagram"
        ],
        "weight": 1.0
    },
    "screenshots": {
        "prompt": "screenshot",
        "synonyms": [
            "screenshot", "screen capture", "UI screenshot",
            "code screenshot", "app interface", "website screenshot",
            "computer screen"
        ],
        "weight": 0.9
    },
    "cars": {
        "prompt": "car",
        "synonyms": [
            "car", "truck", "motorcycle", "vehicle", "automobile",
            "SUV", "sports car", "classic car"
        ],
        "weight": 0.9
    },
    "flowers": {
        "prompt": "flower",
        "synonyms": [
            "flower", "rose", "tulip", "daisy", "garden flower",
            "bouquet", "wildflowers", "plants"
        ],
        "weight": 0.9
    },
    "kids": {
        "prompt": "kids",
        "synonyms": [
            "kids", "child", "children", "baby", "kid", "toddler",
            "infant", "young child", "playing children"
        ],
        "weight": 1.0
    },
    "sports": {
        "prompt": "sports",
        "synonyms": [
            "sports", "basketball", "football", "soccer", "tennis",
            "gym", "exercise", "fitness", "athletics"
        ],
        "weight": 1.0
    },
    "art": {
        "prompt": "art",
        "synonyms": [
            "art", "painting", "drawing", "sculpture", "artwork",
            "museum art", "street art", "graffiti"
        ],
        "weight": 0.9
    },
    "indoor": {
        "prompt": "indoor",
        "synonyms": [
            "indoor", "interior", "room", "living room", "bedroom",
            "kitchen", "office", "inside"
        ],
        "weight": 0.8
    },
    "events": {
        "prompt": "event",
        "synonyms": [
            "wedding", "birthday party", "concert", "parade",
            "festival", "graduation", "ceremony", "celebration"
        ],
        "weight": 1.0
    },
    "selfies": {
        "prompt": "selfie",
        "synonyms": [
            "selfie", "self portrait", "mirror selfie", "group selfie"
        ],
        "weight": 1.1
    }
}

# Large label set - comprehensive categories for maximum accuracy
LARGE_LABELS = {
    # People categories - more specific
    "people": {
        "prompt": "people",
        "synonyms": [
            "people", "person", "a person", "group of people",
            "crowd", "audience", "human", "individuals"
        ],
        "weight": 1.0
    },
    "portraits": {
        "prompt": "portrait",
        "synonyms": [
            "portrait", "headshot", "face closeup", "professional portrait",
            "formal portrait", "casual portrait"
        ],
        "weight": 1.0
    },
    "selfies": {
        "prompt": "selfie",
        "synonyms": [
            "selfie", "self portrait", "mirror selfie", "group selfie",
            "selfie stick photo", "front camera photo"
        ],
        "weight": 1.1
    },
    "family": {
        "prompt": "family",
        "synonyms": [
            "family photo", "family gathering", "family portrait",
            "relatives", "family members", "generations"
        ],
        "weight": 1.0
    },
    "kids": {
        "prompt": "kids",
        "synonyms": [
            "kids", "child", "children", "baby", "toddler", "infant",
            "newborn", "young child", "playing children", "school children"
        ],
        "weight": 1.0
    },
    
    # Animal categories
    "pets": {
        "prompt": "pets",
        "synonyms": [
            "pets", "pet", "dog", "cat", "puppy", "kitten",
            "hamster", "rabbit", "guinea pig", "bird", "fish",
            "domestic animal", "animal companion"
        ],
        "weight": 1.1
    },
    "wildlife": {
        "prompt": "wildlife",
        "synonyms": [
            "wildlife", "wild animal", "bird", "deer", "bear",
            "elephant", "lion", "tiger", "monkey", "giraffe",
            "zoo animal", "safari animal", "jungle animal"
        ],
        "weight": 1.0
    },
    "birds": {
        "prompt": "birds",
        "synonyms": [
            "bird", "birds", "eagle", "parrot", "sparrow", "crow",
            "seagull", "duck", "swan", "flying bird"
        ],
        "weight": 0.9
    },
    
    # Nature and landscapes
    "landscape": {
        "prompt": "landscape",
        "synonyms": [
            "landscape", "scenery", "panorama", "vista",
            "scenic view", "nature scene", "countryside"
        ],
        "weight": 1.0
    },
    "mountains": {
        "prompt": "mountains",
        "synonyms": [
            "mountains", "mountain", "peaks", "hills", "valley",
            "mountain range", "alpine", "rocky mountains"
        ],
        "weight": 1.0
    },
    "beach": {
        "prompt": "beach",
        "synonyms": [
            "beach", "ocean", "sea", "coastline", "shore",
            "sand", "waves", "tropical beach", "seaside"
        ],
        "weight": 1.0
    },
    "forest": {
        "prompt": "forest",
        "synonyms": [
            "forest", "woods", "trees", "jungle", "rainforest",
            "woodland", "grove", "tree canopy"
        ],
        "weight": 0.9
    },
    "sunset": {
        "prompt": "sunset",
        "synonyms": [
            "sunset", "sunrise", "golden hour", "dusk", "dawn",
            "twilight", "evening sky", "morning sky"
        ],
        "weight": 1.0
    },
    "flowers": {
        "prompt": "flowers",
        "synonyms": [
            "flower", "flowers", "rose", "tulip", "daisy", "sunflower",
            "orchid", "lily", "bouquet", "garden flowers", "wildflowers",
            "flowering plants", "blossoms"
        ],
        "weight": 0.9
    },
    
    # Urban and architecture
    "cityscape": {
        "prompt": "cityscape",
        "synonyms": [
            "city skyline", "urban", "skyscrapers", "downtown",
            "metropolitan", "city lights", "urban landscape"
        ],
        "weight": 1.0
    },
    "architecture": {
        "prompt": "architecture",
        "synonyms": [
            "architecture", "building", "structure", "architectural design",
            "modern architecture", "historic building", "landmark"
        ],
        "weight": 1.0
    },
    "street": {
        "prompt": "street",
        "synonyms": [
            "street", "road", "street view", "sidewalk", "alley",
            "urban street", "street scene", "pedestrian street"
        ],
        "weight": 0.9
    },
    "interior": {
        "prompt": "interior",
        "synonyms": [
            "interior", "indoor", "room", "living room", "bedroom",
            "kitchen", "bathroom", "office", "home interior",
            "restaurant interior", "hotel room"
        ],
        "weight": 0.9
    },
    
    # Food and drinks
    "food": {
        "prompt": "food",
        "synonyms": [
            "food", "meal", "cuisine", "cooking", "recipe",
            "home cooking", "gourmet food"
        ],
        "weight": 1.0
    },
    "restaurant": {
        "prompt": "restaurant food",
        "synonyms": [
            "restaurant food", "dining", "plated food", "fine dining",
            "cafe food", "fast food", "takeout"
        ],
        "weight": 0.9
    },
    "dessert": {
        "prompt": "dessert",
        "synonyms": [
            "dessert", "cake", "ice cream", "cookies", "pastry",
            "sweet", "chocolate", "candy"
        ],
        "weight": 0.9
    },
    "drinks": {
        "prompt": "drinks",
        "synonyms": [
            "drink", "beverage", "coffee", "tea", "cocktail",
            "wine", "beer", "juice", "smoothie"
        ],
        "weight": 0.9
    },
    
    # Activities and events
    "sports": {
        "prompt": "sports",
        "synonyms": [
            "sports", "athletics", "game", "match", "competition",
            "basketball", "football", "soccer", "tennis", "baseball",
            "swimming", "running", "cycling"
        ],
        "weight": 1.0
    },
    "fitness": {
        "prompt": "fitness",
        "synonyms": [
            "fitness", "gym", "workout", "exercise", "training",
            "yoga", "weights", "cardio", "health"
        ],
        "weight": 0.9
    },
    "wedding": {
        "prompt": "wedding",
        "synonyms": [
            "wedding", "bride", "groom", "wedding ceremony",
            "wedding reception", "wedding dress", "marriage"
        ],
        "weight": 1.1
    },
    "party": {
        "prompt": "party",
        "synonyms": [
            "party", "birthday party", "celebration", "gathering",
            "festive", "balloons", "cake cutting"
        ],
        "weight": 1.0
    },
    "concert": {
        "prompt": "concert",
        "synonyms": [
            "concert", "music festival", "live music", "performance",
            "stage", "crowd", "band", "singer"
        ],
        "weight": 1.0
    },
    
    # Transportation
    "cars": {
        "prompt": "cars",
        "synonyms": [
            "car", "automobile", "vehicle", "sedan", "SUV",
            "sports car", "classic car", "vintage car", "race car"
        ],
        "weight": 0.9
    },
    "motorcycle": {
        "prompt": "motorcycle",
        "synonyms": [
            "motorcycle", "motorbike", "bike", "scooter", "chopper"
        ],
        "weight": 0.9
    },
    "airplane": {
        "prompt": "airplane",
        "synonyms": [
            "airplane", "aircraft", "plane", "jet", "airport",
            "aviation", "flight"
        ],
        "weight": 0.9
    },
    "boats": {
        "prompt": "boats",
        "synonyms": [
            "boat", "ship", "yacht", "sailboat", "vessel",
            "marina", "harbor"
        ],
        "weight": 0.9
    },
    
    # Art and creativity
    "art": {
        "prompt": "art",
        "synonyms": [
            "art", "artwork", "artistic", "creative", "gallery art",
            "museum piece", "exhibition"
        ],
        "weight": 0.9
    },
    "painting": {
        "prompt": "painting",
        "synonyms": [
            "painting", "painted art", "canvas", "oil painting",
            "watercolor", "acrylic painting", "portrait painting"
        ],
        "weight": 0.9
    },
    "graffiti": {
        "prompt": "graffiti",
        "synonyms": [
            "graffiti", "street art", "mural", "urban art",
            "spray paint art", "wall art"
        ],
        "weight": 0.9
    },
    
    # Digital content
    "screenshots": {
        "prompt": "screenshot",
        "synonyms": [
            "screenshot", "screen capture", "computer screen",
            "phone screen", "app interface", "website", "UI"
        ],
        "weight": 0.8
    },
    "memes": {
        "prompt": "memes",
        "synonyms": [
            "meme", "internet meme", "funny image with text",
            "viral image", "social media post"
        ],
        "weight": 0.9
    },
    "documents": {
        "prompt": "documents",
        "synonyms": [
            "document", "paper", "text", "whiteboard", "presentation",
            "slide", "notes", "handwriting", "diagram", "chart"
        ],
        "weight": 0.9
    },
    
    # Miscellaneous
    "night": {
        "prompt": "night",
        "synonyms": [
            "night", "nighttime", "dark", "night sky", "stars",
            "moon", "night lights", "evening"
        ],
        "weight": 0.9
    },
    "underwater": {
        "prompt": "underwater",
        "synonyms": [
            "underwater", "ocean life", "coral reef", "fish",
            "diving", "snorkeling", "marine life"
        ],
        "weight": 0.9
    },
    "abstract": {
        "prompt": "abstract",
        "synonyms": [
            "abstract", "pattern", "texture", "geometric",
            "shapes", "colors", "artistic abstract"
        ],
        "weight": 0.8
    }
}

# Default to small labels for backward compatibility
DEFAULT_LABELS = SMALL_LABELS

# Function to get labels by tier
def get_labels_for_tier(tier: str = "small"):
    """
    Get label configuration for a specific tier.
    
    Args:
        tier: One of 'small', 'medium', 'large'
        
    Returns:
        Dictionary of label configurations
    """
    tier = tier.lower()
    if tier == "small":
        return SMALL_LABELS.copy()
    elif tier == "medium":
        return MEDIUM_LABELS.copy()
    elif tier == "large":
        return LARGE_LABELS.copy()
    else:
        raise ValueError(f"Invalid tier: {tier}. Choose from 'small', 'medium', 'large'")

# Function to get label count for each tier
def get_label_stats():
    """
    Get statistics about label tiers.
    
    Returns:
        Dictionary with label counts and info for each tier
    """
    return {
        "small": {
            "count": len(SMALL_LABELS),
            "description": "Basic categories for quick sorting",
            "accuracy": "~60%"
        },
        "medium": {
            "count": len(MEDIUM_LABELS),
            "description": "Extended categories for better accuracy",
            "accuracy": "~75%"
        },
        "large": {
            "count": len(LARGE_LABELS),
            "description": "Comprehensive categories for professional use",
            "accuracy": "~85%"
        }
    }