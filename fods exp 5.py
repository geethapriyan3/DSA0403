import numpy as np

# Fuel efficiency of different car models (in MPG)
fuel_efficiency = np.array([25, 30, 35, 40, 45])

# Calculate average fuel efficiency
average = np.mean(fuel_efficiency)

# Compare Model 1 and Model 2
model1 = fuel_efficiency[0]   # 25 MPG
model2 = fuel_efficiency[1]   # 30 MPG

# Calculate percentage improvement
percentage_improvement = ((model2 - model1) / model1) * 100

# Display results
print("Average fuel efficiency:", average, "MPG")
print("Percentage improvement:", percentage_improvement, "%")

