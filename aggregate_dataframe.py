def transform_and_select_columns_dynamic(df:pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate and transform columns with +/- suffixes, returning mean (avg) and std metrics per group.

    For each 'idx' group, this function:
    1. Computes the mean of 'prob' column
    2. For each column with +/- suffixes, calculates mean (renamed to 'avg_') and standard deviation ('std_')
    3. Returns a DataFrame with one row per 'idx' containing aggregated metrics

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing:
        - 'idx' column (grouping key)
        - 'prob' column (will be averaged)
        - Columns with +/- suffixes (e.g., 'Smax+', 'M2Total-') to be aggregated

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with:
        - One row per unique 'idx' value
        - Columns: 'idx', 'prob' (averaged), and aggregated metrics ('avg_*', 'std_*')
        - Original +/- columns are excluded from output

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'idx': [1, 1, 2, 2],
    ...     'prob': [0.1, 0.3, 0.5, 0.7],
    ...     'Smax+': [10, 20, 30, 40],
    ...     'nBonds-': [2, 4, 6, 8]
    ... })
    >>> result = transform_and_select_columns_dynamic(data)
    >>> print(result)
       idx  prob  avg_Smax+  std_Smax+  avg_nBonds-  std_nBonds-
    0    1   0.2       15.0   7.071068          3.0     1.414214
    1    2   0.6       35.0   7.071068          7.0     1.414214

    Notes
    -----
    - The function automatically handles any columns matching the pattern: {base_columns}{suffixes}
    - Base columns processed: ['Smax', 'Smax2', 'Smax3', 'Smax4', 'M2Total', 'M2Prime', 'nBonds']
    - Suffixes processed: ['+', '-']
    - Missing columns are silently skipped
    """
    suffixes = ['+', '-']
    base_columns = ['Smax', 'Smax2', 'Smax3', 'Smax4','M2Total','M2Prime','nBonds']
    
    # Columns to aggregate (prob and all +/- columns)
    agg_cols = ['prob']
    for base in base_columns:
        for suffix in suffixes:
            col = f"{base}{suffix}"
            if col in df.columns:
                agg_cols.append(col)
    
    # Compute mean (avg) and std
    agg_results = df.groupby('idx')[agg_cols].agg(['mean', 'std'])
    
    # Flatten columns and rename 'mean_' to 'avg_'
    new_columns = []
    for col, stat in agg_results.columns:
        if stat == 'mean':
            new_columns.append(f'avg_{col}')
        elif stat == 'std':
            new_columns.append(f'std_{col}')
        else:
            new_columns.append(col)
    agg_results.columns = new_columns
    
    # Rename 'avg_prob' back to 'prob' (if needed)
    if 'avg_prob' in agg_results.columns:
        agg_results = agg_results.rename(columns={'avg_prob': 'prob'})
    
    # Reset index to keep 'idx' as a column
    agg_results.reset_index(inplace=True)
    
    # Select desired columns (idx, prob, avg_*, std_*)
    column_order = ['idx', 'prob']
    for base in base_columns:
        for suffix in suffixes:
            col = f"{base}{suffix}"
            if f'avg_{col}' in agg_results.columns:
                column_order.extend([f'avg_{col}', f'std_{col}'])
    
    return agg_results[column_order]