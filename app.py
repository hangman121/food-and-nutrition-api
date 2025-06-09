# from flask import Flask, request, jsonify
# from PIL import Image
# import torch
# import torch.nn.functional as F
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# import io
# import base64
# import json
# import requests
# import hashlib
# import hmac
# import time
# import urllib.parse
# from typing import Dict, List, Optional
# import os
# from dotenv import load_dotenv

# # Load .env file variables
# load_dotenv()


# # Initialize Flask app
# app = Flask(__name__)

# # FatSecret API Class
# class FatSecretAPI:
#     def __init__(self, consumer_key: str, consumer_secret: str):
#         self.consumer_key = consumer_key
#         self.consumer_secret = consumer_secret
#         self.base_url = "https://platform.fatsecret.com/rest/server.api"
    
#     def _generate_signature(self, method: str, url: str, params: Dict[str, str]) -> str:
#         """Generate OAuth 1.0 signature for FatSecret API"""
#         sorted_params = sorted(params.items())
#         param_string = '&'.join([f"{k}={urllib.parse.quote(str(v), safe='')}" 
#                                 for k, v in sorted_params])
#         signature_base = f"{method}&{urllib.parse.quote(url, safe='')}&{urllib.parse.quote(param_string, safe='')}"
#         signing_key = f"{urllib.parse.quote(self.consumer_secret, safe='')}&"
#         signature = hmac.new(
#             signing_key.encode(),
#             signature_base.encode(),
#             hashlib.sha1
#         ).digest()
#         return base64.b64encode(signature).decode()
    
#     def _make_request(self, params: Dict[str, str]) -> Dict:
#         """Make authenticated request to FatSecret API"""
#         oauth_params = {
#             'oauth_consumer_key': self.consumer_key,
#             'oauth_nonce': str(int(time.time())),
#             'oauth_signature_method': 'HMAC-SHA1',
#             'oauth_timestamp': str(int(time.time())),
#             'oauth_version': '1.0'
#         }
#         all_params = {**params, **oauth_params}
#         signature = self._generate_signature('POST', self.base_url, all_params)
#         all_params['oauth_signature'] = signature
        
#         response = requests.post(self.base_url, data=all_params)
#         if response.status_code == 200:
#             return response.json()
#         else:
#             raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
#     def search_foods(self, search_expression: str, max_results: int = 1) -> List[Dict]:
#         """Search for foods by name"""
#         params = {
#             'method': 'foods.search',
#             'search_expression': search_expression,
#             'max_results': str(max_results),
#             'format': 'json'
#         }
#         try:
#             result = self._make_request(params)
#             if 'foods' in result and 'food' in result['foods']:
#                 foods = result['foods']['food']
#                 if isinstance(foods, dict):
#                     foods = [foods]
#                 return foods
#             else:
#                 return []
#         except Exception as e:
#             print(f"Error searching foods: {e}")
#             return []
    
#     def get_food_details(self, food_id: str) -> Optional[Dict]:
#         """Get detailed nutrition information for a specific food"""
#         params = {
#             'method': 'food.get.v2',
#             'food_id': food_id,
#             'format': 'json'
#         }
#         try:
#             result = self._make_request(params)
#             return result.get('food', {})
#         except Exception as e:
#             print(f"Error getting food details: {e}")
#             return None
    
#     def get_food_nutrition_json(self, food_name: str) -> Dict:
#         """Get nutrition information for the top search result"""
#         foods = self.search_foods(food_name, max_results=1)
        
#         if not foods:
#             return {"error": f"No foods found for '{food_name}'"}
        
#         top_food = foods[0]
#         food_id = top_food.get('food_id')
        
#         if not food_id:
#             return {"error": "Could not get food ID"}
        
#         nutrition_data = self.get_food_details(food_id)
        
#         if not nutrition_data:
#             return {"error": "Could not get nutrition details"}
        
#         return self._format_nutrition_info(nutrition_data)
    
#     def _format_nutrition_info(self, food_data: Dict) -> Dict:
#         """Format nutrition information into a readable structure (top serving only)"""
#         info = {
#             'food_name': food_data.get('food_name', 'Unknown'),
#             'food_type': food_data.get('food_type', 'Unknown'),
#             'food_url': food_data.get('food_url', ''),
#             'serving': {}
#         }
        
#         servings_data = food_data.get('servings', {})
#         if 'serving' in servings_data:
#             servings = servings_data['serving']
#             if isinstance(servings, dict):
#                 first_serving = servings
#             else:
#                 first_serving = servings[0] if servings else {}
            
#             info['serving'] = {
#                 'serving_description': first_serving.get('serving_description', ''),
#                 'metric_serving_amount': first_serving.get('metric_serving_amount', ''),
#                 'metric_serving_unit': first_serving.get('metric_serving_unit', ''),
#                 'measurement_description': first_serving.get('measurement_description', ''),
#                 'calories': first_serving.get('calories', '0'),
#                 'carbohydrate': first_serving.get('carbohydrate', '0'),
#                 'protein': first_serving.get('protein', '0'),
#                 'fat': first_serving.get('fat', '0'),
#                 'saturated_fat': first_serving.get('saturated_fat', '0'),
#                 'polyunsaturated_fat': first_serving.get('polyunsaturated_fat', '0'),
#                 'monounsaturated_fat': first_serving.get('monounsaturated_fat', '0'),
#                 'trans_fat': first_serving.get('trans_fat', '0'),
#                 'cholesterol': first_serving.get('cholesterol', '0'),
#                 'sodium': first_serving.get('sodium', '0'),
#                 'potassium': first_serving.get('potassium', '0'),
#                 'fiber': first_serving.get('fiber', '0'),
#                 'sugar': first_serving.get('sugar', '0'),
#                 'vitamin_a': first_serving.get('vitamin_a', '0'),
#                 'vitamin_c': first_serving.get('vitamin_c', '0'),
#                 'calcium': first_serving.get('calcium', '0'),
#                 'iron': first_serving.get('iron', '0')
#             }
        
#         return info

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_NAME = "skylord/swin-finetuned-food101"

# # Global variables
# processor = None
# model = None
# fs_api = None

# def load_models():
#     """Load the Swin Transformer model and initialize FatSecret API"""
#     global processor, model, fs_api

#     print("🔄 Loading food recognition model...")
#     processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
#     model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
#     model.to(device).eval()
#     print("✅ Swin Transformer model loaded successfully!")

#     # Initialize FatSecret API
#     CONSUMER_KEY = os.getenv("FATSECRET_CONSUMER_KEY")
#     CONSUMER_SECRET = os.getenv("FATSECRET_CONSUMER_SECRET")
#     fs_api = FatSecretAPI(CONSUMER_KEY, CONSUMER_SECRET)
#     print("✅ FatSecret API initialized!")

# def predict_food(image):
#     """Predict food from PIL image using the Swin Transformer model"""
#     inputs = processor(images=image, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     probs = F.softmax(outputs.logits, dim=-1)[0]
#     top_prob, top_idx = torch.max(probs, dim=0)
#     label = model.config.id2label[top_idx.item()]
#     return label, top_prob.item()

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({"status": "healthy", "message": "Food Nutrition API is running"})

# @app.route('/predict', methods=['POST'])
# def predict_nutrition():
#     """Main endpoint for food recognition and nutrition analysis"""
#     try:
#         # Check if image is provided
#         if 'image' not in request.files:
#             return jsonify({"error": "No image file provided"}), 400
        
#         file = request.files['image']
#         if file.filename == '':
#             return jsonify({"error": "No image file selected"}), 400
        
#         # Load and process image
#         try:
#             image = Image.open(file.stream).convert("RGB")
#         except Exception as e:
#             return jsonify({"error": f"Invalid image file: {str(e)}"}), 400
        
#         # Predict food
#         food_name, confidence = predict_food(image)
        
#         # Get nutrition information
#         nutrition_info = fs_api.get_food_nutrition_json(food_name)
        
#         # Prepare response
#         response = {
#             "prediction": {
#                 "food_name": food_name,
#                 "confidence": round(confidence, 4)
#             },
#             "nutrition": nutrition_info
#         }
        
#         return jsonify(response)
    
#     except Exception as e:
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# @app.route('/predict-base64', methods=['POST'])
# def predict_nutrition_base64():
#     """Alternative endpoint for base64 encoded images"""
#     try:
#         data = request.get_json()
        
#         if not data or 'image' not in data:
#             return jsonify({"error": "No base64 image data provided"}), 400
        
#         # Decode base64 image
#         try:
#             image_data = base64.b64decode(data['image'])
#             image = Image.open(io.BytesIO(image_data)).convert("RGB")
#         except Exception as e:
#             return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400
        
#         # Predict food
#         food_name, confidence = predict_food(image)
        
#         # Get nutrition information
#         nutrition_info = fs_api.get_food_nutrition_json(food_name)
        
#         # Prepare response
#         response = {
#             "prediction": {
#                 "food_name": food_name,
#                 "confidence": round(confidence, 4)
#             },
#             "nutrition": nutrition_info
#         }
        
#         return jsonify(response)
    
#     except Exception as e:
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# @app.route('/', methods=['GET'])
# def home():
#     """Home endpoint with API documentation"""
#     docs = {
#         "message": "Food Recognition & Nutrition API",
#         "endpoints": {
#             "POST /predict": "Upload image file for food recognition and nutrition analysis",
#             "POST /predict-base64": "Send base64 encoded image for analysis",
#             "GET /health": "Health check endpoint"
#         },
#         "usage": {
#             "file_upload": "Send POST request to /predict with 'image' field containing image file",
#             "base64": "Send POST request to /predict-base64 with JSON: {'image': 'base64_string'}"
#         }
#     }
#     return jsonify(docs)

# # Always load models when app is imported (e.g. by gunicorn)
# load_models()

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=False)


from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
import io
import base64
import json
import requests
import hashlib
import hmac
import time
import urllib.parse
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# Load .env file variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# FatSecret API Class (unchanged)
class FatSecretAPI:
    def __init__(self, consumer_key: str, consumer_secret: str):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.base_url = "https://platform.fatsecret.com/rest/server.api"
    
    def _generate_signature(self, method: str, url: str, params: Dict[str, str]) -> str:
        """Generate OAuth 1.0 signature for FatSecret API"""
        sorted_params = sorted(params.items())
        param_string = '&'.join([f"{k}={urllib.parse.quote(str(v), safe='')}" 
                                for k, v in sorted_params])
        signature_base = f"{method}&{urllib.parse.quote(url, safe='')}&{urllib.parse.quote(param_string, safe='')}"
        signing_key = f"{urllib.parse.quote(self.consumer_secret, safe='')}&"
        signature = hmac.new(
            signing_key.encode(),
            signature_base.encode(),
            hashlib.sha1
        ).digest()
        return base64.b64encode(signature).decode()
    
    def _make_request(self, params: Dict[str, str]) -> Dict:
        """Make authenticated request to FatSecret API"""
        oauth_params = {
            'oauth_consumer_key': self.consumer_key,
            'oauth_nonce': str(int(time.time())),
            'oauth_signature_method': 'HMAC-SHA1',
            'oauth_timestamp': str(int(time.time())),
            'oauth_version': '1.0'
        }
        all_params = {**params, **oauth_params}
        signature = self._generate_signature('POST', self.base_url, all_params)
        all_params['oauth_signature'] = signature
        
        response = requests.post(self.base_url, data=all_params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
    def search_foods(self, search_expression: str, max_results: int = 1) -> List[Dict]:
        """Search for foods by name"""
        params = {
            'method': 'foods.search',
            'search_expression': search_expression,
            'max_results': str(max_results),
            'format': 'json'
        }
        try:
            result = self._make_request(params)
            if 'foods' in result and 'food' in result['foods']:
                foods = result['foods']['food']
                if isinstance(foods, dict):
                    foods = [foods]
                return foods
            else:
                return []
        except Exception as e:
            print(f"Error searching foods: {e}")
            return []
    
    def get_food_details(self, food_id: str) -> Optional[Dict]:
        """Get detailed nutrition information for a specific food"""
        params = {
            'method': 'food.get.v2',
            'food_id': food_id,
            'format': 'json'
        }
        try:
            result = self._make_request(params)
            return result.get('food', {})
        except Exception as e:
            print(f"Error getting food details: {e}")
            return None
    
    def get_food_nutrition_json(self, food_name: str) -> Dict:
        """Get nutrition information for the top search result"""
        foods = self.search_foods(food_name, max_results=1)
        
        if not foods:
            return {"error": f"No foods found for '{food_name}'"}
        
        top_food = foods[0]
        food_id = top_food.get('food_id')
        
        if not food_id:
            return {"error": "Could not get food ID"}
        
        nutrition_data = self.get_food_details(food_id)
        
        if not nutrition_data:
            return {"error": "Could not get nutrition details"}
        
        return self._format_nutrition_info(nutrition_data)
    
    def _format_nutrition_info(self, food_data: Dict) -> Dict:
        """Format nutrition information into a readable structure (top serving only)"""
        info = {
            'food_name': food_data.get('food_name', 'Unknown'),
            'food_type': food_data.get('food_type', 'Unknown'),
            'food_url': food_data.get('food_url', ''),
            'serving': {}
        }
        
        servings_data = food_data.get('servings', {})
        if 'serving' in servings_data:
            servings = servings_data['serving']
            if isinstance(servings, dict):
                first_serving = servings
            else:
                first_serving = servings[0] if servings else {}
            
            info['serving'] = {
                'serving_description': first_serving.get('serving_description', ''),
                'metric_serving_amount': first_serving.get('metric_serving_amount', ''),
                'metric_serving_unit': first_serving.get('metric_serving_unit', ''),
                'measurement_description': first_serving.get('measurement_description', ''),
                'calories': first_serving.get('calories', '0'),
                'carbohydrate': first_serving.get('carbohydrate', '0'),
                'protein': first_serving.get('protein', '0'),
                'fat': first_serving.get('fat', '0'),
                'saturated_fat': first_serving.get('saturated_fat', '0'),
                'polyunsaturated_fat': first_serving.get('polyunsaturated_fat', '0'),
                'monounsaturated_fat': first_serving.get('monounsaturated_fat', '0'),
                'trans_fat': first_serving.get('trans_fat', '0'),
                'cholesterol': first_serving.get('cholesterol', '0'),
                'sodium': first_serving.get('sodium', '0'),
                'potassium': first_serving.get('potassium', '0'),
                'fiber': first_serving.get('fiber', '0'),
                'sugar': first_serving.get('sugar', '0'),
                'vitamin_a': first_serving.get('vitamin_a', '0'),
                'vitamin_c': first_serving.get('vitamin_c', '0'),
                'calcium': first_serving.get('calcium', '0'),
                'iron': first_serving.get('iron', '0')
            }
        
        return info

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "skylord/swin-finetuned-food101"

# Global variables
processor = None
model = None
fs_api = None

def load_models():
    """Load the Swin Transformer model and initialize FatSecret API"""
    global processor, model, fs_api

    print("🔄 Loading food recognition model...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.to(device).eval()
    print("✅ Swin Transformer model loaded successfully!")

    # Initialize FatSecret API
    CONSUMER_KEY = os.getenv("FATSECRET_CONSUMER_KEY")
    CONSUMER_SECRET = os.getenv("FATSECRET_CONSUMER_SECRET")
    fs_api = FatSecretAPI(CONSUMER_KEY, CONSUMER_SECRET)
    print("✅ FatSecret API initialized!")

def preprocess_image(image):
    """Preprocess image to ensure compatibility with the model"""
    try:
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image if it's too large (optional but recommended)
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Ensure minimum size (some models require minimum dimensions)
        min_size = 224
        if min(image.size) < min_size:
            # Calculate new size maintaining aspect ratio
            width, height = image.size
            if width < height:
                new_width = min_size
                new_height = int((height * min_size) / width)
            else:
                new_height = min_size
                new_width = int((width * min_size) / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

def predict_food(image):
    """Predict food from PIL image using the Swin Transformer model"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        print(f"Processed image - Mode: {processed_image.mode}, Size: {processed_image.size}")
        
        # Alternative approach: Use the processor with minimal parameters
        try:
            # Method 1: Simple processing
            inputs = processor(processed_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            try:
                # Method 2: With padding
                inputs = processor([processed_image], return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
                # Method 3: Manual preprocessing
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                pixel_values = transform(processed_image).unsqueeze(0).to(device)
                inputs = {"pixel_values": pixel_values}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=-1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)
        label = model.config.id2label[top_idx.item()]
        
        # Clean up GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return label, top_prob.item()
        
    except Exception as e:
        print(f"Error in predict_food: {e}")
        print(f"Image mode: {image.mode}, size: {image.size}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "message": "Food Nutrition API is running",
        "model_loaded": processor is not None and model is not None
    })

@app.route('/predict', methods=['POST'])
def predict_nutrition():
    """Main endpoint for food recognition and nutrition analysis"""
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Load and process image
        try:
            image = Image.open(file.stream)
            print(f"Original image mode: {image.mode}, size: {image.size}")
        except Exception as e:
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400
        
        # Predict food
        try:
            food_name, confidence = predict_food(image)
            print(f"Predicted: {food_name} with confidence: {confidence}")
        except Exception as e:
            return jsonify({"error": f"Food prediction failed: {str(e)}"}), 500
        
        # Get nutrition information
        nutrition_info = fs_api.get_food_nutrition_json(food_name)
        
        # Prepare response
        response = {
            "prediction": {
                "food_name": food_name,
                "confidence": round(confidence, 4)
            },
            "nutrition": nutrition_info
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Unexpected error in predict_nutrition: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/predict-base64', methods=['POST'])
def predict_nutrition_base64():
    """Alternative endpoint for base64 encoded images"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No base64 image data provided"}), 400
        
        # Decode base64 image
        try:
            # Handle data URL format (data:image/jpeg;base64,...)
            image_data_str = data['image']
            if image_data_str.startswith('data:image'):
                # Remove data URL prefix
                image_data_str = image_data_str.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(image_data_str)
            image = Image.open(io.BytesIO(image_data))
            print(f"Base64 image mode: {image.mode}, size: {image.size}")
            
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400
        
        # Predict food
        try:
            food_name, confidence = predict_food(image)
            print(f"Predicted: {food_name} with confidence: {confidence}")
        except Exception as e:
            return jsonify({"error": f"Food prediction failed: {str(e)}"}), 500
        
        # Get nutrition information
        nutrition_info = fs_api.get_food_nutrition_json(food_name)
        
        # Prepare response
        response = {
            "prediction": {
                "food_name": food_name,
                "confidence": round(confidence, 4)
            },
            "nutrition": nutrition_info
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Unexpected error in predict_nutrition_base64: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    docs = {
        "message": "Food Recognition & Nutrition API",
        "endpoints": {
            "POST /predict": "Upload image file for food recognition and nutrition analysis",
            "POST /predict-base64": "Send base64 encoded image for analysis",
            "GET /health": "Health check endpoint"
        },
        "usage": {
            "file_upload": "Send POST request to /predict with 'image' field containing image file",
            "base64": "Send POST request to /predict-base64 with JSON: {'image': 'base64_string'}"
        },
        "model_info": {
            "status": "loaded" if processor is not None and model is not None else "not_loaded",
            "model_name": MODEL_NAME,
            "device": str(device)
        }
    }
    return jsonify(docs)

# Always load models when app is imported (e.g. by gunicorn)
load_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)