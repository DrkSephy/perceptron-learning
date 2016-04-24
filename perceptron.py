from random import choice
from numpy import array, dot, random

# Create unit step function
unit_step = lambda x: 0 if x < 0 else 1

# Map input to output
# First two values in the array are the inputs
# Third value is the bias
# Last value is the expected output
training_data = [
  (array([0, 0, 1]), 0),
  (array([0, 1, 1]), 1),
  (array([1, 0, 1]), 1),
  (array([1, 1, 1]), 1),
]

# Initialize the weights randomly
w = random.rand(3)

# Learning rate alpha
alpha = 0.2

# Number of iterations
iterations = 0

# While we have not converged...
while True:
  # Store error count
  error_count = 0

  # Increment Iteration count
  iterations += 1

  # Loop over training data
  for x, expected in training_data:

    # Compute dot product
    result = dot(x, w)

    # Compute the error
    error = expected - unit_step(result)

    # If there was en error, update the weights
    if error != 0:
      # Increment error count by 1
      error_count += 1
      w += alpha * error * x

  # If there were no weight changes for the entire epoch, we are done training
  if error_count == 0:
    print '{} {} {} : {}'.format('Converged in', str(iterations) + ' iterations', 'with weights', w)
    break

# At this point, our perceptrons should have learned the logical OR function
for x, _ in training_data:
  result = dot(x, w)
  print '{}: {} -> {}'.format(x[:2], result, unit_step(result))