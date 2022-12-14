import requests
import json

# URL for the web service, should be similar to:

scoring_uri = ''

# If the service is authenticated, set the key or token

key = ''

data = {"data":
        [
          {
           "age": 50, 
           "anaemia": 1, 
           "creatinine_phosphokinase": 168, 
           "diabetes": 0, 
           "ejection_fraction": 38, 
           "high_blood_pressure": 1, 
           "platelets": 276000, 
           "serum_creatinine": 1.1, 
           "serum_sodium": 137, 
           "sex": 1, 
           "smoking": 0,
           "time": 11
          },
          {
           "age": 49, 
           "anaemia": 1, 
           "creatinine_phosphokinase": 80, 
           "diabetes": 0, 
           "ejection_fraction": 30, 
           "high_blood_pressure": 1, 
           "platelets": 427000, 
           "serum_creatinine": 1, 
           "serum_sodium": 138, 
           "sex": 0, 
           "smoking": 0,
           "time": 12
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

