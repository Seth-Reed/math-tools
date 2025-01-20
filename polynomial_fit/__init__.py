import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import minimize


def increment(term, max_value):
    """
    Increment a term in the polynomial term list.

    Parameters:
        term (list): current term to be incremented.
        max_value (int): the maximum value that any element in the term can possess.

    Returns:
        list or None: The incremented term or None if incrementing is no longer possible.
    """

    # loop through term backwards
    for i in range(len(term)-1, -1, -1):
        if term[i] < max_value:
            term[i] += 1  # increment if we have not reached last variable
            for j in range(i+1, len(term)):
                term[j] = term[i]  # increment each term value after the current to avoid repeating terms ([0, 1, 0] becomes [0, 1, 1], since [0, 1, 0] = [0, 0, 1])
            return term
    # end external while loop if we have incremented as much as possible
    return None


def generate_terms(num_var, order):
    """
    Generate all possible terms for a polynomial with given number of variables and order.

    Parameters:
        num_var (int): the number of input variables in the polynomial.
        order (int): the order of the polynomial.

    Returns:
        list: a list of terms, where each term is represented by a list of exponents.

        e.g. term list for 2nd order two-variable polynomial = [[], [0], [1], [0, 0], [0, 1], [1, 1]]
    """
    terms = []
    for r in range(order + 1):
        term = [0] * r  # generate base term format
        while term is not None:
            terms.append(term.copy())  # add the initial term/incremented term
            term = increment(term, max_value=num_var-1)
    return terms


def polynomial_function(coeffs, X, terms):
    """
    Calculate the polynomial function value for a given set of inputs.

    Parameters:
        coeffs (ndarray): coefficients of the polynomial.
        X (ndarray): input values for the polynomial.
                        If there's more than 1 input, then each row of X corresponds to a different input.
        terms (list): list of terms in the polynomial.

    Returns:
        y (ndarray): the calculated polynomial function values.
    """
    X = np.atleast_2d(X)
    n = X.shape[0]
    m = X.shape[1]

    y = np.zeros(m)

    for coeff, term in zip(coeffs, terms):
        powers = [term.count(i) for i in range(n)]
        term_value = np.prod([np.power(X[i, :], power) for i, power in enumerate(powers)], axis=0)
        y += coeff * term_value
    return y


def residual_function(coeffs, X, y, terms):
    """
    Compute the residuals of the polynomial function given the observed data.
    """
    polynomial_value = polynomial_function(coeffs, X, terms)
    return y - polynomial_value


def objective_function(coeffs, X, y, terms):
    """
    Objective function for the optimization, computing the sum of squared residuals.
    """
    res = residual_function(coeffs, X, y, terms)
    return np.sum(res ** 2)


def constraint_function(coeffs, X_constraints, y_constraints, terms):
    """
    Constraint function for the optimization.
    """
    polynomial_constraint = polynomial_function(coeffs, X_constraints, terms)
    return np.array(y_constraints - polynomial_constraint)


def polynomial_fit(input_data, output_data, order, input_constraints=None, output_constraints=None):
    """
    Fit a polynomial to the given data, optionally subject to constraints.

    Parameters:
        input_data (ndarray): input data for the polynomial fitting,
                              each row should be its own input.
        output_data (ndarray): observed output data for the polynomial fitting.
        order (int): the order of the desired polynomial fit.
        input_constraints (ndarray, optional): input values for any constraints.
                                               each row should be its own input.
        output_constraints (ndarray, optional): output values for any constraints.

    Returns:
        tuple: a tuple containing the optimal polynomial coefficients and the corresponding list of terms.
    """
    X = np.atleast_2d(input_data)
    y = output_data
    n = X.shape[0]
    r = order

    # generate term combinations and initial guess for coefficients
    terms = generate_terms(num_var=n, order=r)
    initial_guess = np.ones(len(terms))

    # formulate constraint and objective functions
    if input_constraints is None or output_constraints is None:
        constraint = None
    else:
        constraint = ({'type': 'eq', 'fun': lambda coeffs: constraint_function(coeffs,
                                                                               input_constraints,
                                                                               output_constraints,
                                                                               terms)})
    objective = partial(objective_function, X=X, y=y, terms=terms)

    # optimize polynomial
    result = minimize(fun=objective,
                      x0=initial_guess,
                      constraints=constraint)
    optimal_coeffs = list(result.x)
    # compute R^2
    r_squared = compute_r_squared(y_true=y,
                                  y_pred=polynomial_function(coeffs=optimal_coeffs,
                                                             X=X,
                                                             terms=terms))
    rsme = compute_rmse(y_true=y,
                        y_pred=polynomial_function(coeffs=optimal_coeffs,
                                                   X=X,
                                                   terms=terms))

    metrics = [r_squared, rsme]

    return optimal_coeffs, terms, metrics


def compute_r_squared(y_true, y_pred):
    """
    Compute the coefficient of determination (R^2) for the polynomial fit.

    Parameters:
        y_true (array_like): true output values.
        y_pred (array_like): predicted output values by the polynomial

    Returns:
        float: R^2 value
    """
    tss = np.sum((y_true - np.mean(y_true)) ** 2)   # Total Sum of Squares
    rss = np.sum((y_true - y_pred) ** 2)            # Residual Sum of Squares
    r_squared = 1 - (rss / tss)
    return r_squared


def compute_rmse(y_true, y_pred):
    """
    Compute the Root Mean Square Error (RMSE) for the model predictions.

    Parameters:
        y_true (array_like): The true output values.
        y_pred (array_like): The predicted output values by the model.

    Returns:
        float: RMSE value
    """
    mse = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    return rmse


def polynomial_to_string(terms, coefficients, variables):
    """
    Convert a list of terms and coefficients to a multivariate polynomial string representation.

    Parameters:
    terms (list): list of terms, where each term is a list of powers for each variable.
    coefficients (list): list of coefficients corresponding to each term.
    variables (list): A list of variable names as strings.

    Returns:
    str: A string representation of the multivariate polynomial.
    """
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")   # superscript map
    poly_str = ''

    for term, coeff in zip(terms, coefficients):
        if coeff == 0:      # ignore coefficients = 0
            continue

        # add +/- sign
        if poly_str:        # check if string is not empty
            poly_str += ' + ' if coeff > 0 else ' - '
        elif coeff < 0:     # string is empty and coefficient is negative
            poly_str += '-'
            coeff = abs(coeff)

        # add coefficient
        coeff = abs(coeff)
        if coeff != 1 or not term:  # if coeff is not 1 or it's a constant
            poly_str += f'{coeff:.8f}'

        # construct term based on index frequency
        for i in set(term):
            count = term.count(i)   # count number of occurrences
            if count > 0:
                poly_str += variables[i]  # add variable
                if count > 1:
                    poly_str += str(count).translate(superscripts)  # add power

    return poly_str


def differentiate_polynomial(coeffs, terms, x):
    """
    Differentiate a polynomial with respect to a particular variable.

    Parameters:
    coefficients (list): list of coefficients corresponding to each term.
    terms (list): list of terms, where each term is a list of powers for each variable.
    x (list): A list pointing to which variable to differentiate the function with respect to.
                [Ex: x = [0, 0, 1] would differentiate a 3 variable polynomial with respect to its third variable]

    Returns:
    tuple: a tuple containing the optimal polynomial coefficients and the corresponding list of terms.
    """
    derivative_coefficients = []
    derivative_terms = []

    n = len(x)
    v = x.index(1)

    for coeff, term in zip(coeffs, terms):
        power = term.count(v)

        if power > 0:
            derivative_coefficient = power * coeff
            index_to_pop = sum([term.count(i) for i in range(v)])
            term.pop(index_to_pop)
            derivative_coefficients.append(float(derivative_coefficient))
            derivative_terms.append(term)


    return derivative_coefficients, derivative_terms


def one_D_example():
    def example_1D_function(x):
        y = np.sin(2.*np.pi*x / 5.)
        return y

    # Generate example data
    x_data = np.linspace(0., 5., 40)
    y_data = example_1D_function(x_data)

    # Define Constraint(s)
    x_constraints = np.array([0.])
    y_constraints = np.array([0.])  # these force the polynomial to pass through the origin

    # Fit Polynomial
    order = 5
    optimal_coeffs, terms, r_squared = polynomial_fit(input_data=x_data,
                                                      output_data=y_data,
                                                      order=order,
                                                      input_constraints=x_constraints,
                                                      output_constraints=y_constraints)

    # Print Results
    # print('1-Dimensional Polynomial Fit Results')
    # print(f'Terms = {terms}')
    # print(f'Coefficients = {optimal_coeffs}')
    # print(f"R^2 = {r_squared}")
    # print(f'Polynomial Fit: y = ' + polynomial_to_string(terms, optimal_coeffs, ['x']))
    # print()

    derivative_coeffs, derivative_terms = differentiate_polynomial(optimal_coeffs, terms, x=[1])

    # Get Points for Plotting
    x_range = np.linspace(x_data[0], x_data[-1], 100)
    y_fit = polynomial_function(optimal_coeffs, x_range, terms)

    # Plot for Comparison
    plt.figure()
    plt.plot(x_data, y_data, 'bx', label='$y = sin(x)$')
    plt.plot(x_range, y_fit, 'r', label='Polynomial Fit')
    plt.plot(x_constraints, y_constraints, 'gx', label='Constraints')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('1-Dimensional Polynomial Fit Example')


def two_D_example():
    def example_2D_function(x1, x2):
        y = np.sin(x1)*np.cos(x2)
        return y

    # Generate example data
    x1_data = np.random.uniform(-1, 1, 100)
    x2_data = np.random.uniform(-1, 1, 100)
    X_data = np.vstack((x1_data, x2_data))
    y_data = example_2D_function(x1_data, x2_data)

    # Define Constraint(s)
    x1_constraints = np.array([0])
    x2_constraints = np.array([0])
    X_constraints = np.vstack((x1_constraints, x2_constraints))
    y_constraints = np.array([0])

    # Fit Polynomial
    order = 4
    optimal_coeffs, terms, r_squared = polynomial_fit(input_data=X_data,
                                                      output_data=y_data,
                                                      order=order,
                                                      input_constraints=X_constraints,
                                                      output_constraints=y_constraints)

    derivative_coeffs, derivative_terms = differentiate_polynomial(optimal_coeffs, terms, x=[0, 1])

    # Print Results
    print('2-Dimensional Polynomial Fit Results')
    print(f'Terms = {terms}')
    print(f'Coefficients = {optimal_coeffs}')
    print(f"R^2 = {r_squared}")
    print(f'Polynomial Fit: y = ' + polynomial_to_string(terms, optimal_coeffs, ['x₁', 'x₂']))
    print()

    # Get Points for Plotting
    x1_range = np.linspace(-1, 1, 30)
    x2_range = np.linspace(-1, 1, 30)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    X_grid_vector = np.vstack((X1_grid.ravel(), X2_grid.ravel()))
    fit = polynomial_function(optimal_coeffs, X_grid_vector, terms).reshape(X1_grid.shape)

    # Plot for Comparison
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1_data, x2_data, y_data, marker='o', color='b', label='$y = \\sin(x_1)\\cos(x_2)$')
    ax.plot_surface(X1_grid, X2_grid, fit, alpha=0.7, color='r')
    ax.scatter(x1_constraints, x2_constraints, y_constraints, marker='x', color='g', label='Constraints')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$y$')
    ax.set_title('2-Dimensional Polynomial Fit Example')
    ax.legend()


if __name__ == '__main__':
    # one_D_example()
    # two_D_example()
    # plt.show()

    num_var = 0
    order = 16
    terms = generate_terms(num_var, order)
    coeffs = np.random.rand(len(terms))

    print(terms)
    print(coeffs)
    print(polynomial_to_string(terms, coeffs, variables=['x', 'y']))
    print()

    d_coeffs, d_terms = differentiate_polynomial(coeffs, terms, x=[1,0])

    print(d_terms)
    print(d_coeffs)
    print(polynomial_to_string(d_terms, d_coeffs, variables=['x', 'y']))


