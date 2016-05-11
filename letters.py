from random import choice
from numpy import array, dot, random, all

# Unit step function
unit_step = lambda x: -1 if x < 0 else 1

# A, B, C, D, E, J, K
training_data = [
  (array([-1, -1, +1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, +1, +1, +1]), array([+1, -1, -1, -1, -1, -1, -1])),
  (array([+1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, -1]), array([-1, +1, -1, -1, -1, -1, -1])),
  (array([-1, -1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, +1, -1, -1, +1, +1, +1, +1, -1]), array([-1, -1, +1, -1, -1, -1, -1])),
  (array([+1, +1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, +1, +1, -1, -1]), array([-1, -1, -1, +1, -1, -1, -1])),
  (array([+1, +1, +1, +1, +1, +1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1]), array([-1, -1, -1, -1, +1, -1, -1])),
  (array([-1, -1, -1, +1, +1, +1, +1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, -1, -1]), array([-1, -1, -1, -1, -1, +1, -1])),
  (array([+1, +1, +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, -1, +1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1, +1, +1]), array([-1, -1, -1, -1, -1, -1, +1])),
]

# Initialize the weights randomly
w = random.rand(63)

# Learning rate alpha
alpha = 0.2

# Number of iterations
iterations = 0

while True:
  error_count = 0
  iterations += 1

  for x, expected in training_data:
    result = dot(x, w)

    error = expected - unit_step(result)

    if expected.all() != unit_step(result):
      # print error
      # print dot(x, error)
    # if error != 0:
    #   error_count += 1

      # print dot(x, error)
      # w += alpha * error * x

  if error_count == 0:
    print '{} {} {} : {}'.format('Converged in', str(iterations) + ' iterations', 'with weights', w)
    break  

# for x, _ in training_data:
#   result = dot(x, w)
#   print '{}: {} -> {}'.format(x, result, unit_step(result))