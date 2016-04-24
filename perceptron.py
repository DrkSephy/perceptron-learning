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
iterations = 100

# Do the training:
  # 1. Get random input set from training data
  # 2. Calculate dot product of input and weight vectors
  # 3. Correct the weights by multiplying error with learning rate
for i in xrange(iterations):
  # Grab input set and expected output
  x, expected = choice(training_data)

  # Compute dot product
  result = dot(w, x)

  # Compute the error
  error = expected - unit_step(result)

  # Update the weights
  w += alpha * error * x

# At this point, our perceptrons should have learned the logical OR function
for x, _ in training_data:
  result = dot(x, w)
  print '{}: {} -> {}'.format(x[:2], result, unit_step(result))