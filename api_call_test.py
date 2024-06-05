import requests
import os
import json

# URL to which the POST request is to be sent
url = 'https://aidev.automhr.com/api/v1/facial-recognition/create-check-in-out/'  # Replace with the actual URL


# Set the path to the image file on your local device
image_path = '/home/heet/workspace/fr/data/img/VELANKANI_1715342517.203041_6.jpg'
filename = os.path.basename(image_path)

# Set the track ID
location_id = '4'

# Create the payload with track ID
payload = {"email":"heet.b@velankanigroup.com", 'location_id': location_id}

# Send the POST request with the payload and image separately
response = requests.post(url, data=payload, files={'image': open(image_path, 'rb')})


# Printing the status code and response content
print("Status Code:", response.status_code)
response_data = json.loads(response.text)
print(response_data["type"])
# print("Response Body:", )
