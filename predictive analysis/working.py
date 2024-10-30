SCRIPT_REAL(
"
from tabpy.tabpy_tools.client import Client

# Connect to TabPy
client = Client('http://localhost:9004/')

# Hardcoded test data to match your model's expectations
test_data = {
    'Type': [2],  
    'Air temperature K': [303.9],
    'Process temperature K': [313.2],
    'Rotational speed rpm': [1422],
    'Torque Nm': [48],
    'Tool wear min': [215]
}

# Call the model endpoint directly
response = client.query('predict_failure', test_data)
result=response['response']['probabilities'][0][0]
return result
",
ATTR([Type]),ATTR([Air temperature K]),ATTR([Process temperature K]),ATTR([Rotational speed rpm]),ATTR([Torque Nm]),
ATTR([Tool wear min]),ATTR([Target]),
[type],[airtemp],[processtemp],[rotspeed],[torque],[toolwear]
)