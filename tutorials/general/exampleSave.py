import numpy as np
import pickle

# Create a numpy array
test = np.array([[1,2],[3,4]])
test2 = np.array([[5,6], [7,8]])

data = {"A": test, "B": test2}

with open("test_save.pkl", "wb") as output:
    pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

# Now re-load the data and compare
with open("test_save.pkl", "rb") as input:
    loaded_data = pickle.load(input)

print("Original data")
print(f"A is {data['A']}")
print(f"B is {data['B']}")
print("Saved data")
print(f"A is {loaded_data['A']}")
print(f"B is {loaded_data['B']}")