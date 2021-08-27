def portfolio_var(data, conf_coef=1.654):
    """Calculates portfolio correlated VAR
    Input: dataframe of position P&Ls
    Rows: positions; columns: dates; first column must be position ID
    Output: Table of positions and their VAR 
    """
    # conversion of values to numpy array format
    pnl = data.values.astype("float")
    pnl = np.array(pnl).transpose()
    sum_pnl = np.sum(pnl, axis=1, keepdims=True)
    # covariance matrix of each position and total pnl
    cov = np.dot(sum_pnl.T - sum_pnl.mean(), pnl - pnl.mean(axis=0)) / (
        sum_pnl.shape[0] - 1
    )
    cov_coef = cov / np.sum(cov)
    # share of each position in overal portfolio var
    var = (
        np.std(sum_pnl, ddof=1) * conf_coef
    )  # ddof = 1 to calculate STD over sample, not population!
    var_output = (var * cov_coef).reshape(-1, 1)
    var_output = pd.DataFrame(data=var_output, columns=["VAR"], index=data.index)

    return var_output
