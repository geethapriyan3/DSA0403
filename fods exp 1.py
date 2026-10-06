import numpy as np

# Marks of 4 students in the order:
# Math, Science, English, History
student_scores = np.array([
    [80, 75, 90, 85],
    [70, 85, 80, 90],
    [90, 80, 85, 75],
    [85, 90, 95, 80]
])

# Subject names
subjects = ["Math", "Science", "English", "History"]

# Calculate average for each subject (column-wise)
averages = np.mean(student_scores, axis=0)

# Display averages
print("Average score for each subject:")
for subject, average in zip(subjects, averages):
    print(f"{subject}: {average:.2f}")

# Find the subject with the highest average
highest_index = np.argmax(averages)
highest_subject = subjects[highest_index]
highest_average = averages[highest_index]

print("\nSubject with the highest average:")
print(f"{highest_subject}: {highest_average:.2f}")
