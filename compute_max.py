def compute_max(x: pd.Series, y: pd.Series, offset: int, order: int) -> dict:
    """
    Compute the maximum of a function and export errors in x and y.
    
    Args:
        x: x-coordinates (pd.Series)
        y: y-coordinates (pd.Series)
        offset: window size around empirical max for polynomial fit
        order: degree of polynomial fit (e.g., 3 for cubic)
    
    Returns:
        Dictionary containing:
        - Empirical max (x, y)
        - Polynomial max (x, y)
        - Error in x (absolute difference)
        - Error in y (absolute difference)
        - RMSE of the polynomial fit
        - Warnings (if any)
    """
    # Initialize results
    results = {
        'empirical_max_x': x.iloc[y.argmax()],
        'empirical_max_y': y.max(),
        'poly_max_x': None,
        'poly_max_y': None,
        'error_x': None,
        'error_y': None,
        'RMSE': None,
        'warnings': []
    }

    # Extract window around empirical max
    idx_of_max = y.argmax()
    start = max(0, idx_of_max - offset)
    end = min(len(x), idx_of_max + offset + 1)
    x_window = x.iloc[start:end]
    y_window = y.iloc[start:end]

    # Fit polynomial
    coefficients = np.polyfit(x_window, y_window, order)
    poly_func = np.poly1d(coefficients)
    y_fit = poly_func(x_window)
    
    # Compute RMSE
    results['RMSE'] = np.sqrt(np.mean((y_window - y_fit) ** 2))

    # Find critical points (roots of 1st derivative)
    dydx = np.polyder(poly_func)
    critical_points = np.roots(dydx)
    real_critical_points = critical_points[np.isreal(critical_points)].real

    # Check critical points are within bounds
    valid_points = [
        (x_c, poly_func(x_c)) 
        for x_c in real_critical_points 
        if x_window.min() <= x_c <= x_window.max()
    ]

    # Find polynomial maximum (highest valid critical point)
    if valid_points:
        results['poly_max_x'], results['poly_max_y'] = max(valid_points, key=lambda p: p[1])
    else:
        results['warnings'].append("No valid maxima found in polynomial fit")
        return results

    # Compute errors
    results['error_x'] = abs(results['empirical_max_x'] - results['poly_max_x'])
    results['error_y'] = abs(results['empirical_max_y'] - results['poly_max_y'])

    return results