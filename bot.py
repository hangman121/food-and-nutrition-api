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

# Initialize Flask app
app = Flask(__name__)

# FatSecret API Class (same as before)
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
        print(f"Searching nutrition for: {food_name}")
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

# Initialize FatSecret API
CONSUMER_KEY = "2e93840365144042bdfb6ce964f87e6f"
CONSUMER_SECRET = "e6b0331612e749fcb79837e57ae9df76"
fs_api = FatSecretAPI(CONSUMER_KEY, CONSUMER_SECRET)
print("✅ FatSecret API initialized!")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

MODEL_NAME = "skylord/swin-finetuned-food101"

# Load model
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.to(device).eval()
print("✅ Swin Transformer model loaded successfully!")



def predict_food(image):
    """Predict food from PIL image"""
    print("🔍 Analyzing food image...")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)[0]
    top_prob, top_idx = torch.max(probs, dim=0)
    label = model.config.id2label[top_idx.item()]
    print(f"🍕 Predicted: {label} (confidence: {top_prob.item():.4f})")
    return label, top_prob.item()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "message": "Food Nutrition API is running",
        "device": str(device),
        "model_loaded": True
    })

@app.route('/predict', methods=['POST'])
def predict_nutrition():
    """Main endpoint for food recognition and nutrition analysis"""
    print("📨 Received prediction request...")
    
    try:
        # Check if image is provided
        if 'image' not in request.files:
            print("❌ No image file provided")
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            print("❌ No image file selected")
            return jsonify({"error": "No image file selected"}), 400
        
        print(f"📷 Processing image: {file.filename}")
        
        # Load and process image
        try:
            image = Image.open(file.stream).convert("RGB")
            print(f"✅ Image loaded: {image.size}")
        except Exception as e:
            print(f"❌ Invalid image file: {e}")
            return jsonify({"error": f"Invalid image file: {str(e)}"}), 400
        
        # Predict food
        food_name, confidence = predict_food(image)
        
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
        
        print("✅ Response prepared successfully")
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/predict-base64', methods=['POST'])
def predict_nutrition_base64():
    """Alternative endpoint for base64 encoded images"""
    print("📨 Received base64 prediction request...")
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            print("❌ No base64 image data provided")
            return jsonify({"error": "No base64 image data provided"}), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            print(f"✅ Base64 image decoded: {image.size}")
        except Exception as e:
            print(f"❌ Invalid base64 image: {e}")
            return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400
        
        # Predict food
        food_name, confidence = predict_food(image)
        
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
        
        print("✅ Base64 response prepared successfully")
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    docs = {
        "message": "🍕 Food Recognition & Nutrition API - Local Testing",
        "status": "running",
        "device": str(device),
        "endpoints": {
            "POST /predict": "Upload image file for food recognition and nutrition analysis",
            "POST /predict-base64": "Send base64 encoded image for analysis",
            "GET /health": "Health check endpoint"
        },
        "postman_testing": {
            "file_upload": {
                "method": "POST",
                "url": "http://localhost:5000/predict",
                "body_type": "form-data",
                "key": "image",
                "type": "file"
            },
            "base64": {
                "method": "POST", 
                "url": "http://localhost:5000/predict-base64",
                "body_type": "raw (JSON)",
                "example": '{"image": "base64_encoded_string_here"}'
            }
        }
    }
    return jsonify(docs)

if __name__ == '__main__':
    print("🌟 Food Nutrition API is ready!")
    print("🔗 Local URL: http://localhost:5000")
    print("📋 Documentation: http://localhost:5000")
    print("❤️  Health Check: http://localhost:5000/health")
    print("\n" + "="*50)
    
    # Run the app in debug mode for local testing
    app.run(host='0.0.0.0', port=5000, debug=True)