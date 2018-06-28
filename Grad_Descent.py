import numpy as np

#Fit a line of the form y = mx + c using grad descent
def compute_error_for_line_given_points(c, m, points):
    total = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total += (y -(m*x + c)) ** 2 #y is true value, mx + c is value predicted by model
    return total/float(len(points))

def step_gradient(c_curr, m_curr, points, learningRate):
    c_grad = 0
    m_grad = 0
    m = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        c_grad += -(2/m) * (y - ((m_curr*x) + c_curr)) #partial derivative of MSE wrt c
        m_grad += -(2/m) * x * (y - ((m_curr*x)+c_curr)) #[partial derivative of MSE wrt m]
    new_c = c_curr - (learningRate*c_grad)
    new_m = m_curr - (learningRate*m_grad)
    return[new_c,new_m]

def grad_desc_runner(points, starting_c, starting_m, learning_rate, num_iterations):
    c = starting_c
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(c, m, np.array(points), learning_rate)
    return [b, m]

def run(): 
    points = np.genfromtxt("C:/Users/arthu/Documents/Python Scripts/Learn ML/Gradient Descent Straight Line/data.csv", delimiter=",")
    learning_rate = 100
    initial_c = -2e4
    initial_m = 1.43e10
    num_iterations = int(1e5)
    print("Starting gradient descent at c = %g, m = %g, error = %g" \
    % (initial_c, initial_m, compute_error_for_line_given_points(initial_c, initial_m, points)))
    print("Running...")
    [c, m] = grad_desc_runner(points, initial_c, initial_m, learning_rate, num_iterations)
    print("After %d iterations c = %g, m = %g, error = %g" % (num_iterations, c, m, compute_error_for_line_given_points(c, m, points)))
    print("The Young's Modulus for this sample is: %g" % m)

if __name__ == '__main__':
    run()
