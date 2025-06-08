import requests
import base64
import json

# Test the deployed API
API_URL = "http://localhost:5000"  # Change this to your deployed URL

def test_file_upload(image_path):
    """Test with file upload"""
    url = f"{API_URL}/predict"
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(url, files=files)
    
    print("File Upload Test:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_base64_upload(image_path):
    """Test with base64 encoded image"""
    url = f"{API_URL}/predict-base64"
    
    # Convert image to base64
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {"image": image_data}
    response = requests.post(url, json=payload)
    
    print("Base64 Upload Test:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_health_check():
    """Test health check endpoint"""
    url = f"{API_URL}/health"
    response = requests.get(url)
    
    print("Health Check Test:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

if __name__ == "__main__":
    # Test health check
    test_health_check()
    
    # Test with your food image
    image_path = "path/to/your/food_image.jpg"  # Change this path
    
    # Uncomment these lines when you have an image to test
    # test_file_upload(image_path)
    # test_base64_upload(image_path)