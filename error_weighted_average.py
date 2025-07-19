def weighted_average_with_error(df: pd.DataFrame, 
                              property_col: str = 'property', 
                              error_col: str = 'error') -> tuple:
    
    """
    Compute weighted average and its error from a DataFrame.
    
    Args:
        df: Input DataFrame.
        property_col: Name of the property column.
        error_col: Name of the error column.
    
    Returns:
        (weighted_avg, weighted_avg_error)
    """
    weights            = 1 / (df[error_col] ** 2)
    weighted_avg       = np.sum(df[property_col] * weights) / np.sum(weights)
    weighted_avg_error = np.sqrt(1 / np.sum(weights))
    
    return weighted_avg, weighted_avg_error