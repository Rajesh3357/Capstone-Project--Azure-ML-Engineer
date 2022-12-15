import requests
import json

# URL for the web service, should be similar to:

scoring_uri = ''

# If the service is authenticated, set the key or token

key = ''

data = {"data":
        [
          {
           "age": 60, 
           "anaemia": 1, 
           "creatinine_phosphokinase": 315, 
           "diabetes": 1, 
           "ejection_fraction": 60, 
           "high_blood_pressure": 0, 
           "platelets": 454000, 
           "serum_creatinine": 1.1, 
           "serum_sodium": 131, 
           "sex": 1, 
           "smoking": 1,
           "time": 10
          },
          {
           "age": 55, 
           "anaemia": 0, 
           "creatinine_phosphokinase": 1820, 
           "diabetes": 0, 
           "ejection_fraction": 38, 
           "high_blood_pressure": 0, 
           "platelets": 270000, 
           "serum_creatinine": 1.2, 
           "serum_sodium": 139, 
           "sex": 0, 
           "smoking": 0,
           "time": 271
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'


# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
print("Expected result: [true, false], where 'true' means '1' and 'false' means '0' as result in the 'DEATH_EVENT' column")