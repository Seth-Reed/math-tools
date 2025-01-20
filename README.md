# Polynomial Fit Package

This package provides tools for fitting multivariate polynomials to data, optionally subject to constraints. The package includes functionality for generating polynomial terms, fitting data, evaluating the fitted polynomial, computing its analytical derivative, and visualizing the results.

## Features

- Generate polynomial terms up to a specified order.
- Fit multivariate polynomials to data with or without constraints.
- Compute residuals, R-squared values, and RMSE.
- Differentiate multivariate polynomials with respect to specific variables.
- Examples of polynomial fits in one and two dimensions.

---

## Installation

Clone this repository and ensure you have the required dependencies installed.

### Dependencies

- Python 3.6+
- NumPy
- Matplotlib
- SciPy

Install the dependencies with:

```bash
pip install numpy matplotlib scipy
```

---

## Functions

### 1. `generate_terms(num_var, order)`
Generates all possible terms for a polynomial with a given number of variables and order.

#### Parameters:
- `num_var (int)`: Number of variables.
- `order (int)`: Order of the polynomial.

#### Returns:
- `list`: List of terms represented as lists of powers for each variable.

---

### 2. `polynomial_function(coeffs, X, terms)`
Evaluates the polynomial function for given inputs.

#### Parameters:
- `coeffs (ndarray)`: Coefficients of the polynomial.
- `X (ndarray)`: Input data.
- `terms (list)`: List of terms representing the polynomial.

#### Returns:
- `y (ndarray)`: Polynomial function values.

---

### 3. `polynomial_fit(input_data, output_data, order, input_constraints=None, output_constraints=None)`
Fits a polynomial to data with optional constraints.

#### Parameters:
- `input_data (ndarray)`: Input data.
- `output_data (ndarray)`: Observed outputs.
- `order (int)`: Order of the polynomial.
- `input_constraints (ndarray, optional)`: Input constraint values.
- `output_constraints (ndarray, optional)`: Output constraint values.

#### Returns:
- `tuple`: Optimal coefficients, terms, and fit metrics (R² and RMSE).

---

### 4. `differentiate_polynomial(coeffs, terms, x)`
Computes the derivative of the polynomial with respect to a specific variable.

#### Parameters:
- `coeffs (list)`: Coefficients of the polynomial.
- `terms (list)`: List of terms representing the polynomial.
- `x (list)`: A binary list specifying which variable to differentiate.

#### Returns:
- `tuple`: Differentiated coefficients and terms.

---

### 5. `polynomial_to_string(terms, coefficients, variables)`
Converts the polynomial to a human-readable string.

#### Parameters:
- `terms (list)`: List of polynomial terms.
- `coefficients (list)`: Coefficients of the terms.
- `variables (list)`: Variable names as strings.

#### Returns:
- `str`: Polynomial in string format.

---

### 6. `one_D_example()`
Provides an example of a 1D polynomial fit and visualizes the result.

---

### 7. `two_D_example()`
Provides an example of a 2D polynomial fit and visualizes the result in 3D.

---

## Usage

### Fitting a Polynomial

```python
import numpy as np
from polynomial_fit import polynomial_fit, polynomial_to_string

# Define data
x_data = np.linspace(0, 10, 50)
y_data = 2.5 * x_data ** 2 - 1.2 * x_data + 0.8

# Fit polynomial
order = 2
coeffs, terms, metrics = polynomial_fit(x_data, y_data, order)

# Print results
print("Polynomial:", polynomial_to_string(terms, coeffs, ["x"]))
print("Metrics:", metrics)
```

### Differentiating a Polynomial

```python
from polynomial_fit import differentiate_polynomial

derivative_coeffs, derivative_terms = differentiate_polynomial(coeffs, terms, x=[1])
print("Derivative:", polynomial_to_string(derivative_terms, derivative_coeffs, ["x"]))
```

---

## Examples

### 1D Fit Example
Run `one_D_example()` to visualize a 1D polynomial fit.

### 2D Fit Example
Run `two_D_example()` to visualize a 2D polynomial fit with a 3D plot.

---

## Contributing
Feel free to open issues or submit pull requests if you have suggestions for improvement.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
