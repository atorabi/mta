import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import copy
import math
from sklearn.covariance import LedoitWolf, OAS


# Packett Burmann OA with multiple of 4
def choicen(n):
    if n >= 1 and n <= 23:
        degree = n
    else:
        if n > 23 and n <= 31:
            degree = 31
        else:
            if n > 31 and n <= 39:
                degree = 39
            else:
                if n > 39 and n <= 47:
                    degree = 47
                else:
                    if n > 47 and n <= 63:
                        degree = 63
                    else:
                        if n > 63 and n <= 79:
                            degree = 79
                        else:
                            if n > 79 and n <= 95:
                                degree = 95
                            else:
                                if n > 95 and n <= 127:
                                    degree = 127
                                else:
                                    if n > 127 and n <= 319:
                                        degree = 319
                                    else:
                                        if n > 319 and n <= 511:
                                            degree = 511

    return degree


# Packett Burmann Orthogonal Array


def pbdesign(n):
    """
    Generate a Plackett-Burman design

    Parameter
    ---------
    n : int
        The number of factors to create a matrix for.

    Returns
    -------
    H : 2d-array
        An orthogonal design matrix with n columns, one for each factor, and
        the number of rows being the next multiple of 4 higher than n (e.g.,
        for 1-3 factors there are 4 rows, for 4-7 factors there are 8 rows,
        etc.)

    Example
    -------

    A 3-factor design::

        >>> pbdesign(3)
        array([[-1., -1.,  1.],
               [ 1., -1., -1.],
               [-1.,  1., -1.],
               [ 1.,  1.,  1.]])

    A 5-factor design::

        >>> pbdesign(5)
        array([[-1., -1.,  1., -1.,  1.],
               [ 1., -1., -1., -1., -1.],
               [-1.,  1., -1., -1.,  1.],
               [ 1.,  1.,  1., -1., -1.],
               [-1., -1.,  1.,  1., -1.],
               [ 1., -1., -1.,  1.,  1.],
               [-1.,  1., -1.,  1., -1.],
               [ 1.,  1.,  1.,  1.,  1.]])

    """
    assert n > 0, "Number of factors must be a positive integer"
    keep = int(n)
    n = 4 * (int(n / 4) + 1)  # calculate the correct number of rows (multiple of 4)
    f, e = np.frexp([n, n / 12.0, n / 20.0])
    k = [idx for idx, val in enumerate(np.logical_and(f == 0.5, e > 0)) if val]

    assert isinstance(n, int) and k != [], "Invalid inputs. n must be a multiple of 4."

    k = k[0]
    e = e[k] - 1

    if k == 0:  # N = 1*2**e
        H = np.ones((1, 1))
    elif k == 1:  # N = 12*2**e
        H = np.vstack(
            (
                np.ones((1, 12)),
                np.hstack(
                    (
                        np.ones((11, 1)),
                        toeplitz(
                            [-1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1],
                            [-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
                        ),
                    )
                ),
            )
        )
    elif k == 2:  # N = 20*2**e
        H = np.vstack(
            (
                np.ones((1, 20)),
                np.hstack(
                    (
                        np.ones((19, 1)),
                        hankel(
                            [
                                -1,
                                -1,
                                1,
                                1,
                                -1,
                                -1,
                                -1,
                                -1,
                                1,
                                -1,
                                1,
                                -1,
                                1,
                                1,
                                1,
                                1,
                                -1,
                                -1,
                                1,
                            ],
                            [
                                1,
                                -1,
                                -1,
                                1,
                                1,
                                -1,
                                -1,
                                -1,
                                -1,
                                1,
                                -1,
                                1,
                                -1,
                                1,
                                1,
                                1,
                                1,
                                -1,
                                -1,
                            ],
                        ),
                    )
                ),
            )
        )

    # Kronecker product construction
    for i in range(e):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

    # Reduce the size of the matrix as needed
    H = H[:, 1 : (keep + 1)]

    return np.flipud(H)


def OApb(nvar):
    m = choicen(nvar)
    a = pbdesign(m)
    df = pd.DataFrame(np.array(a))
    nDOE = df.shape[0]
    ix = list(range(0, nDOE))  # [0,1,2,3,4]
    iy = list(range(0, nvar))
    dg = df.iloc[ix, iy]
    dg = dg.replace([-1], [2])
    return dg


def OA(nvar):
    c = nvar
    nx = math.log(c, 10) / math.log(2, 10)
    n2 = int(nx) + 1
    # larger runs
    n2 = n2 + 3

    # calling hadamard with power of 2 for the OA
    bh = hadamard(2**n2)
    df = pd.DataFrame(np.array(bh))
    nDOE = 2**n2
    ix = list(range(0, nDOE))  # [0,1,2,3,4]
    iy = list(range(1, nvar + 1))
    dg = df.iloc[ix, iy]
    dg = dg.replace([-1], [2])
    # dg.to_csv('OA.csv',index=False)
    return dg


def runMD(healthydata, sickdata):
    """
    plots mahalanobis distances
    """

    mdHealthy = getMD(healthydata, healthydata)

    dfh = pd.DataFrame(mdHealthy)

    dfh.to_csv("mdheathy.csv", header=None)
    mdSick = getMD(healthydata, sickdata)
    dfs = pd.DataFrame(mdSick)
    dfs.to_csv("mdsick.csv", header=None)


# def standardize(healthydata, data):
#     """standardizes data, based off mean/std of 'healthydata'"""
#     stand = (data - healthydata.mean(axis=0)) / healthydata.std(axis=0, ddof=0)
#     return stand


def standardize(healthydata, data):
    """standardizes data, based off mean/std of 'healthydata'"""
    mu = healthydata.mean(axis=0)
    sigma = healthydata.std(axis=0, ddof=0)
    # protect against zero std
    sigma_safe = np.where(sigma == 0, 1e-12, sigma)
    return (data - mu) / sigma_safe


def getMD(healthydata, data2):
    """
    calculates mahalanobis distance of paramater data2
    based on healthydata
    """
    # Impute the data

    data = standardize(healthydata, data2)  # standardizes matrix
    # print("data", data)
    cov = np.cov(data, rowvar=False)  # covariance
    # print("cov", cov)
    try:
        # inverse = np.linalg.inv(cov)  # takes the inverse of the correlation matrix

        inverse = np.linalg.inv(cov + np.identity(cov.shape[0]) * 1e-6)

        step = np.matmul(data, inverse)  # multiply the inverse with data

        MD = np.matmul(
            step, np.matrix.transpose(data)
        )  # multiply the transpoe(data) with data*inverse
        MD = np.diagonal(MD)
        # MD = np.sqrt(MD / data.shape[1])

        MD = np.nan_to_num(MD, nan=0.0, posinf=np.max(MD[np.isfinite(MD)]), neginf=0.0)
        MD = np.where(MD < 0, 0, MD)
        MD = np.sqrt(MD / data.shape[1])
        print("MD", MD)
    except:
        MD = np.diagonal(np.sqrt(np.matmul(data, np.matrix.transpose(data)) / cov))
        print("MD single", MD)

    return MD  # returns the MDs


# def getMD(healthydata, data2, lam=1e-6, eig_floor=1e-12):
#     """
#     Calculates Mahalanobis distance of param data2 based on healthydata
#     with numerical safeguards:
#       - ridge regularization on covariance
#       - pinv fallback
#       - eigenvalue flooring (optional)
#       - clip tiny negative d^2 before sqrt
#     """
#     # Standardize
#     data = standardize(healthydata, data2)  # shape: (n_samples, n_features)

#     # Covariance on the standardized data
#     cov = np.cov(data, rowvar=False)  # shape: (p, p)

#     # Ensure symmetry (numerical)
#     cov = 0.5 * (cov + cov.T)

#     p = cov.shape[0]

#     # Regularized inverse: (cov + lam*I)^(-1)
#     try:
#         inv_cov = np.linalg.inv(cov + lam * np.eye(p))
#     except np.linalg.LinAlgError:
#         # Fallback 1: eigenvalue flooring + inv
#         w, V = np.linalg.eigh(cov)
#         w = np.maximum(w, eig_floor)
#         cov_reg = (V * w) @ V.T  # V diag(w) V^T
#         try:
#             inv_cov = np.linalg.inv(cov_reg)
#         except np.linalg.LinAlgError:
#             # Fallback 2: Moore–Penrose pseudo-inverse
#             inv_cov = np.linalg.pinv(cov)

#     # d^2 = row-wise quadratic form
#     # (data @ inv_cov) * data, then rowwise sum
#     step = data @ inv_cov
#     d2 = np.einsum("ij,ij->i", step, data)  # diagonal without forming full product

#     # Clip tiny negative values from numerical noise
#     d2 = np.where(d2 < 0, 0.0, d2)

#     # MTS convention: divide by number of features before sqrt
#     MD = np.sqrt(d2 / data.shape[1])

#     return MD


def runMTS(healthydata, sickdata, ortho, colNames):
    """
    runs a full mts
    plots on and off bar graph
    """
    print("ortho", ortho)
    mahalanobisD = []  # empty list for all MD's
    for line in ortho:  # iterates through ortho to determine on off
        rowstoremove = np.where(line == 2)[0]  # gets indices of vars to switch off
        h = copy.deepcopy(healthydata)  # copies data to preserve original
        s = copy.deepcopy(sickdata)  # same as before
        for value in reversed(rowstoremove):  # reverse to not change indices
            s = np.delete(s, value, 1)  # deletes off variables
            h = np.delete(h, value, 1)  # same as before
        print("h", h)
        print("s", s)
        MD = getMD(h, s)  # calculated MD based off run

        mahalanobisD.append(list(MD))  # MDs stored

    TaguchiArray = np.reciprocal(mahalanobisD)  # 1/md^2

    StoN = []  # empty list for signal to noise
    for run in TaguchiArray:  # iterates through each run

        sn = (-10) * math.log((sum(run)), 10)
        StoN.append(sn)
    AverageOn = []  # empty list for on
    for row in np.transpose(ortho):  # tranpos to determine when var was on
        x = 0  # resets x
        rowstoaverage = np.where(row == 1)[0]  # determines where var was on
        for value in rowstoaverage:  # sums values
            x += StoN[value]
        AverageOn.append(x / len(rowstoaverage))  # calculates average
    AverageOff = []  # same process as above but for off values
    for row in np.transpose(ortho):
        x = 0
        rowstoaverage = np.where(row == 2)[0]
        for value in rowstoaverage:
            x += StoN[value]
        AverageOff.append(x / len(rowstoaverage))

    dfon = pd.DataFrame(np.array(AverageOn))
    dfof = pd.DataFrame(np.array(AverageOff))

    dfon.to_csv("SNRon.csv", header=None)
    dfof.to_csv("SNRoff.csv", header=None)

    delta = dfon - dfof
    delta["SNR Gain"] = delta
    delta["Test Parameter"] = colNames

    threshold = 0.0

    # --- Plotting with Plotly ---

    delta = delta.sort_values("SNR Gain", ascending=False)
    delta["Test Recommendation"] = np.where(
        delta["SNR Gain"] < threshold, "Remove", "Keep"
    )

    fig_new = px.bar(
        delta, x="Test Parameter", y="SNR Gain", color="Test Recommendation"
    )

    delta.style.apply(highlight_test)
    delta.to_csv("SNRdelta.csv", index=False)


def highlight_test():
    return ["background-color:yellow"]


def get_numerical_columns_for_md(df: pd.DataFrame):
    """
    Identifies and returns a list of numerical column names based on the presence
    of numerical 'Upperlimit' and 'Lowerlimit' values, and ensuring they are not equal.

    Args:
        df (pd.DataFrame): The original DataFrame.

    Returns:
        list: A list of column names that should be used for MD calculation.
    """
    # Find the rows for Upperlimit and Lowerlimit
    identifier_col = df.columns[0]

    try:
        upperlimit_row = df[df[identifier_col] == "Upperlimit"].iloc[0]
        lowerlimit_row = df[df[identifier_col] == "Lowerlimit"].iloc[0]
    except IndexError:
        st.error(
            "Could not find 'Upperlimit' or 'Lowerlimit' rows. Please check your CSV."
        )
        return []

    numerical_cols = []
    # Iterate through all columns to check if they have numerical limits
    for col in df.columns:
        # Skip essential or known non-numerical columns
        if col in [
            "serial_id",
            "start_time",
            "test_result",
            "build_config",
            "operator",
            "station",
            "fixture_id",
            "cycle_time",
            "image_name_ver",
            "build",
            "route_check",
            "Errorcode",
            "Units",
        ]:
            continue
        try:
            # Check if values in both limit rows can be converted to float
            val_upper = float(upperlimit_row[col])
            val_lower = float(lowerlimit_row[col])

            # Use the user's correct filtering logic: check for non-equal, non-NaN values
            if (val_upper != val_lower) and (
                not np.isnan(val_upper) and not np.isnan(val_lower)
            ):
                numerical_cols.append(col)
        except (ValueError, IndexError, TypeError):
            # If conversion fails, this column is not numerical for our purposes
            continue

    return numerical_cols


# def calculate_mta_anomaly_scores(
#     pass_numerical_data: pd.DataFrame, fail_numerical_data: pd.DataFrame
# ):
#     """
#     Calculates the Multi Test Anomaly (MTA) score for each row in
#     both PASS and FAIL dataframes, based on the PASS data distribution.

#     Args:
#         pass_numerical_data (pd.DataFrame): DataFrame of 'PASS' devices with numerical data only.
#         fail_numerical_data (pd.DataFrame): DataFrame of 'FAIL' devices with numerical data only.

#     Returns:
#         tuple: A tuple containing the calculated scores for pass and fail devices,
#                the calculated threshold, and the average score.
#     """
#     # 1. Impute missing values with the median of each column for the PASS data
#     st.info("Imputing missing values with the median for the baseline 'PASS' data.")
#     pass_imputed_data = pass_numerical_data.fillna(pass_numerical_data.median()).copy()

#     # 2. STANDARDIZATION
#     st.info("Standardizing data using a scaler trained ONLY on the 'PASS' data.")
#     scaler = StandardScaler()
#     scaled_pass_df = pd.DataFrame(
#         scaler.fit_transform(pass_imputed_data), columns=pass_imputed_data.columns
#     )

#     # 3. Calculate the inverse covariance matrix from the SCALED PASS data
#     try:
#         covariance_matrix = np.cov(scaled_pass_df.values, rowvar=False)
#         inv_covariance_matrix = np.linalg.inv(
#             covariance_matrix + np.identity(covariance_matrix.shape[0]) * 1e-6
#         )
#     except np.linalg.LinAlgError:
#         st.error(
#             "Error: The covariance matrix is singular and cannot be inverted. This can happen if columns are perfectly correlated."
#         )
#         return [], [], 0, 0

#     # 4. Calculate Anomaly Score (Mahalanobis distance) for each row in both dataframes
#     st.info("Calculating Anomaly Score for all PASS and FAIL devices.")
#     mean_vector = scaled_pass_df.mean(axis=0).values

#     # Calculate score for PASS devices
#     mta_scores_pass = [
#         mahalanobis(row, mean_vector, inv_covariance_matrix)
#         for _, row in scaled_pass_df.iterrows()
#     ]

#     # Initialize the fail distances list before the conditional block
#     mta_scores_fail = []

#     # Check if there are any FAIL devices before trying to transform them
#     if not fail_numerical_data.empty:
#         st.info("Found FAIL devices. Calculating their Anomaly Scores.")
#         fail_imputed_data = fail_numerical_data.fillna(
#             pass_numerical_data.median()
#         ).copy()
#         scaled_fail_df = pd.DataFrame(
#             scaler.transform(fail_imputed_data), columns=fail_imputed_data.columns
#         )
#         mta_scores_fail = [
#             mahalanobis(row, mean_vector, inv_covariance_matrix)
#             for _, row in scaled_fail_df.iterrows()
#         ]
#     else:
#         st.warning("No 'FAIL' devices found in the dataset. Skipping FAIL analysis.")

#     # 5. Normalize for the Mahalanobis-Taguchi System (MTS)
#     k = pass_imputed_data.shape[1]
#     mts_mahalanobis_distances_pass = np.array(mta_scores_pass) / np.sqrt(k)
#     mts_mahalanobis_distances_fail = np.array(mta_scores_fail) / np.sqrt(k)

#     # Calculate the threshold and average score
#     chi_square_limit = chi2.ppf(0.97, df=k)
#     mta_threshold = chi_square_limit / np.sqrt(k)
#     avg_mta_score = np.mean(mts_mahalanobis_distances_pass)

#     return (
#         mts_mahalanobis_distances_pass,
#         mts_mahalanobis_distances_fail,
#         mta_threshold,
#         avg_mta_score,
#     )


def calculate_mta_anomaly_scores(
    pass_numerical_data: pd.DataFrame,
    fail_numerical_data: pd.DataFrame,
    cov_estimator: str = "LedoitWolf",  # "LedoitWolf" | "OAS" | "Empirical"
    threshold_mode: str = "chi2",  # "chi2" | "empirical"
    empirical_quantile: float = 0.97,  # used when threshold_mode == "empirical"
):
    """
    Calculates MTA scores using a shrinkage covariance estimator for stability in high dimensions.
    Returns (pass_scores, fail_scores, threshold, avg_pass_score).
    """
    # 1) Impute with PASS medians (baseline)
    st.info("Imputing missing values with the median for the baseline 'PASS' data.")
    pass_imputed = pass_numerical_data.fillna(pass_numerical_data.median()).copy()

    # 2) Standardize using only PASS data
    st.info("Standardizing data using a scaler trained ONLY on the 'PASS' data.")
    scaler = StandardScaler()
    X_pass = pd.DataFrame(
        scaler.fit_transform(pass_imputed), columns=pass_imputed.columns
    )

    # 3) Choose covariance estimator (shrinkage preferred)
    st.info(f"Fitting covariance estimator: {cov_estimator}.")
    est = None
    try:
        if cov_estimator == "LedoitWolf":
            est = LedoitWolf(store_precision=True).fit(X_pass.values)
        elif cov_estimator == "OAS":
            est = OAS(store_precision=True).fit(X_pass.values)
        else:  # "Empirical" (fallback to np.cov)
            # Keep your original behavior (less stable)
            cov = np.cov(X_pass.values, rowvar=False)
            # Regularize slightly to avoid singularity
            cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(cov.shape[0])
            precision = np.linalg.pinv(
                cov
            )  # pinv is safer than inv for high-d problems
    except Exception as e:
        st.warning(f"{cov_estimator} failed: {e}. Falling back to OAS.")
        est = OAS(store_precision=True).fit(X_pass.values)

    if est is not None:
        precision = est.precision_  # already the inverse covariance

    # 4) Mahalanobis scores (normalized by sqrt(k)) for PASS
    mean_vec = X_pass.mean(axis=0).values
    diff_pass = X_pass.values - mean_vec
    # quadratic form per row: diag( (X - mu) * P * (X - mu)^T )
    d2_pass = np.einsum("ij,jk,ik->i", diff_pass, precision, diff_pass)
    d2_pass = np.where(d2_pass < 0, 0, d2_pass)  # numerical safety
    k = X_pass.shape[1]
    mta_pass = np.sqrt(d2_pass / k)

    # 5) FAIL (if any)
    mta_fail = []
    if not fail_numerical_data.empty:
        fail_imputed = fail_numerical_data.fillna(pass_numerical_data.median()).copy()
        X_fail = pd.DataFrame(
            scaler.transform(fail_imputed), columns=fail_imputed.columns
        )
        diff_fail = X_fail.values - mean_vec
        d2_fail = np.einsum("ij,jk,ik->i", diff_fail, precision, diff_fail)
        d2_fail = np.where(d2_fail < 0, 0, d2_fail)
        mta_fail = np.sqrt(d2_fail / k)
    else:
        st.warning("No 'FAIL' devices found; skipping FAIL scoring.")

    # 6) Threshold
    if threshold_mode == "empirical":
        # 97th percentile of PASS distances (robust to k changes)
        thr = float(np.quantile(mta_pass, empirical_quantile))
        st.info(
            f"Empirical threshold at {int(empirical_quantile*100)}th percentile of PASS: {thr:.4f}"
        )
    else:
        # classic chi-square-based threshold
        # chi_square_limit = chi2.ppf(0.97, df=k)
        # thr = chi_square_limit / np.sqrt(k)

        # k = pass_imputed_data.shape[1]
        chi_square_limit = chi2.ppf(0.97, df=k)
        thr = np.sqrt(chi_square_limit / k)  # <-- correct for sqrt(d^2 / k)
        st.info(f"Chi-square(0.97, df={k}) threshold: {thr:.4f}")

    avg_pass = float(np.mean(mta_pass))
    return mta_pass, np.array(mta_fail), thr, avg_pass


def real_failure_filter(df):
    """
    Filters the DataFrame to identify and keep only the latest test result
    for each serial_id, classifying a device as a "real" failure only if its final
    test result is 'FAIL'.

    Args:
        df (pd.DataFrame): The original DataFrame containing all test results.

    Returns:
        tuple: A tuple containing the filtered pass and fail DataFrames.
    """
    st.subheader("Identifying 'Real' Failures")
    st.info("Filtering to keep only the latest test result for each device.")

    # Ensure start_time is in a sortable format
    df["start_time"] = pd.to_datetime(
        df["start_time"], format="%Y%m%d_%H%M%S", errors="coerce"
    )

    # Drop any rows where start_time couldn't be parsed
    df.dropna(subset=["start_time"], inplace=True)

    # Sort by serial_id and then start_time
    df_sorted = df.sort_values(by=["serial_id", "start_time"])

    # Get the last (latest) test result for each serial_id
    latest_results = df_sorted.drop_duplicates(subset=["serial_id"], keep="last")

    # Separate into final PASS and final FAIL dataframes
    final_pass_df = latest_results[latest_results["test_result"] == "PASS"].copy()
    final_fail_df = latest_results[latest_results["test_result"] != "PASS"].copy()

    st.write(f"Total Unique Devices Analyzed: {len(latest_results)}")
    st.write(f"Devices with Final 'PASS' Status: {len(final_pass_df)}")
    st.write(f"Devices with Final 'FAIL' Status: {len(final_fail_df)}")
    st.write("---")

    return final_pass_df, final_fail_df


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Multi Test Anomaly (MTA) Analysis")
    st.info(
        "Upload a CSV file to perform a Multi Test Anomaly analysis on 'PASS' tests."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio, low_memory=False)

        numerical_cols_for_md = get_numerical_columns_for_md(df)
        essential_columns = [
            "serial_id",
            "start_time",
            "test_result",
            "build_config",
            "fixture_id",
            "route_check",
        ]
        cols_to_keep = essential_columns + numerical_cols_for_md

        try:
            # Drop the metadata rows (Upperlimit, Lowerlimit, etc.) and filter columns
            filtered_df = df[cols_to_keep].iloc[4:].copy()
        except KeyError as e:
            st.error(
                f"Error: Missing one of the required columns in the uploaded file: {e}."
            )
            return

        # --- Filter based on route_check == 1 ---
        st.subheader("Initial Data Filtering")
        st.info("Filtering dataset to include only rows where 'route_check' is '1'.")
        # Added a check for both string and float to make the code more robust
        filtered_df = filtered_df[
            (filtered_df["route_check"] == "1") | (filtered_df["route_check"] == 1.0)
        ].copy()

        if filtered_df.empty:
            st.warning(
                "No rows found where 'route_check' is '1'. Please check your data."
            )
            return

        # --- Filter data to get real failures and final passes ---
        pass_df, fail_df = real_failure_filter(filtered_df)

        if pass_df.empty:
            st.warning(
                "No rows with 'test_result' equal to 'PASS' found after filtering."
            )
            return

        st.subheader("MTA Score Calculation Parameters")

        # Create numerical-only dataframes for calculation
        pass_numerical_data = pass_df[numerical_cols_for_md].copy()
        fail_numerical_data = fail_df[numerical_cols_for_md].copy()

        # Convert to numeric, coercing any errors to NaN
        for col in pass_numerical_data.columns:
            pass_numerical_data[col] = pd.to_numeric(
                pass_numerical_data[col], errors="coerce"
            )
        for col in fail_numerical_data.columns:
            fail_numerical_data[col] = pd.to_numeric(
                fail_numerical_data[col], errors="coerce"
            )

        # Perform MTA score calculation
        (
            mta_scores_pass,
            mta_scores_fail,
            mta_threshold,
            avg_mta_score,
        ) = calculate_mta_anomaly_scores(pass_numerical_data, fail_numerical_data)

        if len(mta_scores_pass) > 0:
            # Add the scores back to the main dataframes
            pass_df["mta_anomaly_score"] = mta_scores_pass
            fail_df["mta_anomaly_score"] = mta_scores_fail

            # Identify outliers in both datasets
            pass_df["is_outlier"] = pass_df["mta_anomaly_score"] > mta_threshold
            fail_df["is_outlier"] = fail_df["mta_anomaly_score"] > mta_threshold

            # Create a simple label for plotting
            pass_df["test_label"] = "PASS"
            fail_df["test_label"] = "FAIL"

            # Combine for plotting
            combined_df = pd.concat([pass_df, fail_df], ignore_index=True)

            # Ground-truth: anything not PASS is FAIL (this matches your real_failure_filter logic)
            combined_df["gt_fail"] = (
                combined_df["test_result"].astype(str).str.strip().str.upper() != "PASS"
            )

            # MTA classification from scores
            combined_df["mta_result"] = np.where(
                combined_df["mta_anomaly_score"] <= mta_threshold, "PASS", "FAIL"
            )

            # Now compute errors using the ground-truth flag (not the raw string)
            false_negatives = combined_df[
                (combined_df["gt_fail"]) & (combined_df["mta_result"] == "PASS")
            ]
            false_positives = combined_df[
                (~combined_df["gt_fail"]) & (combined_df["mta_result"] == "FAIL")
            ]

            total_original_pass = (~combined_df["gt_fail"]).sum()
            total_original_fail = (combined_df["gt_fail"]).sum()

            false_positive_rate = (
                (len(false_positives) / total_original_pass * 100)
                if total_original_pass
                else 0
            )
            false_negative_rate = (
                (len(false_negatives) / total_original_fail * 100)
                if total_original_fail
                else 0
            )
            # --- Calculation of False Positive & False Negative ---
            # Define MTA-based classification
            combined_df["mta_result"] = np.where(
                combined_df["mta_anomaly_score"] <= mta_threshold, "PASS", "FAIL"
            )

            # # Calculate counts
            # false_positives = combined_df[
            #     (combined_df["test_result"] == "PASS")
            #     & (combined_df["mta_result"] == "FAIL")
            # ]
            # false_negatives = combined_df[
            #     (combined_df["test_result"] == "FAIL")
            #     & (combined_df["mta_result"] == "PASS")
            # ]

            # Calculate rates
            total_original_pass = len(combined_df[combined_df["test_result"] == "PASS"])
            total_original_fail = len(combined_df[combined_df["test_result"] == "FAIL"])

            false_positive_rate = (
                (len(false_positives) / total_original_pass) * 100
                if total_original_pass > 0
                else 0
            )
            false_negative_rate = (
                (len(false_negatives) / total_original_fail) * 100
                if total_original_fail > 0
                else 0
            )

            st.write("---")
            st.subheader("MTA Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="False Positives (FP)", value=len(false_positives))
                st.write(f"FP Rate: **{false_positive_rate:.2f}%**")
            with col2:
                st.metric(label="False Negatives (FN)", value=len(false_negatives))
                st.write(f"FN Rate: **{false_negative_rate:.2f}%**")
            st.write("---")

            st.write(
                f"95% Confidence Upper Limit for MTA Anomaly Score: {mta_threshold:.2f}"
            )
            st.write(f"Average Anomaly Score (PASS Devices): {avg_mta_score:.2f}")

            # Display processed dataframes
            st.subheader("Processed Data with MTA Anomaly Score")
            st.write("---")
            st.markdown("#### Final PASS Devices")
            st.dataframe(pass_df)

            if not fail_df.empty:
                st.markdown("#### Real 'FAIL' Devices")
                st.dataframe(fail_df)
            else:
                st.info("No 'FAIL' devices to display.")

            st.write("---")

            # Define the color map for the plots
            color_discrete_map = {"PASS": "green", "FAIL": "red"}

            # Add Y-axis scale selection
            st.subheader("Visualization Options")
            y_axis_scale = st.radio(
                "Y-axis Scale:",
                ["Linear", "Log"],
                index=0,
                help="Select the scale for the Y-axis (Anomaly Score). Log scale is useful for visualizing points with very small values.",
            )

            # Get the min and max MD for setting the log scale range
            min_mta = combined_df["mta_anomaly_score"].min()
            max_mta = combined_df["mta_anomaly_score"].max()

            # Plot 1: MTA Anomaly Score vs. Start Time
            st.subheader("MTA Anomaly Score vs. Start Time")
            fig1 = px.scatter(
                combined_df,
                x="start_time",
                y="mta_anomaly_score",
                color="test_label",
                symbol="is_outlier",
                color_discrete_map=color_discrete_map,
                title="MTA Anomaly Score over Time",
                hover_data=["serial_id", "build_config", "fixture_id"],
            )
            fig1.add_hline(
                y=mta_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {mta_threshold:.2f}",
            )
            fig1.add_hline(
                y=avg_mta_score,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Average Score: {avg_mta_score:.2f}",
            )
            if y_axis_scale == "Log":
                # Add a small epsilon to the min value to avoid log(0) issues
                y_min_log = np.log10(max(1e-6, min_mta))
                y_max_log = np.log10(max_mta)
                fig1.update_yaxes(type="log", range=[y_min_log, y_max_log])
            st.plotly_chart(fig1)

            # Plot 2: Histogram of MTA Anomaly Score
            st.subheader("Histogram of MTA Anomaly Score")
            fig2 = px.histogram(
                combined_df,
                x="mta_anomaly_score",
                color="test_label",
                nbins=50,
                color_discrete_map=color_discrete_map,
                title="Distribution of MTA Anomaly Score",
            )
            fig2.add_vline(
                x=mta_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {mta_threshold:.2f}",
            )
            fig2.add_vline(
                x=avg_mta_score,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Average Score: {avg_mta_score:.2f}",
            )
            st.plotly_chart(fig2)

            # --- Violin Plot Section ---
            st.write("---")
            st.subheader("MTA Anomaly Score by Categorical Variable")
            categorical_variable = st.selectbox(
                "Select a categorical variable to plot:",
                options=["build_config", "fixture_id"],
                index=0,
            )

            # Check if the selected column exists in the DataFrame
            if categorical_variable in combined_df.columns:
                fig3 = px.violin(
                    combined_df,
                    x=categorical_variable,
                    y="mta_anomaly_score",
                    points="all",
                    color="test_label",
                    color_discrete_map=color_discrete_map,
                    title=f"MTA Anomaly Score vs. {categorical_variable}",
                    hover_data=["serial_id", "build_config", "fixture_id"],
                )
                fig3.add_hline(
                    y=mta_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold: {mta_threshold:.2f}",
                )
                if y_axis_scale == "Log":
                    # Add a small epsilon to the min value to avoid log(0) issues
                    y_min_log = np.log10(max(1e-6, min_mta))
                    y_max_log = np.log10(max_mta)
                    fig3.update_yaxes(type="log", range=[y_min_log, y_max_log])
                st.plotly_chart(fig3)
            else:
                st.warning(
                    f"The column '{categorical_variable}' was not found in the dataset."
                )
            # --- End of Violin Plot Section ---

            st.success("Analysis complete! All outliers have been identified.")

            opt = 1  # Packett Burmann Orthogonal Array

            # passdata = pass_df[numerical_cols_for_md].copy()
            # faildata = fail_df[numerical_cols_for_md].copy()

            # # Convert to numeric, coercing any errors to NaN
            # for col in passdata.columns:
            #     passdata[col] = pd.to_numeric(passdata[col], errors="coerce")
            # for col in faildata.columns:
            #     faildata[col] = pd.to_numeric(faildata[col], errors="coerce")

            # # opt=0 # Read from a csv file
            # ncol = len(passdata.columns)

            # pass_imputed_data = passdata.fillna(passdata.median()).copy()
            # fail_imputed_data = faildata.fillna(passdata.median()).copy()
            # print("fail data", faildata)
            # passdata = pass_imputed_data.to_numpy()
            # faildata = fail_imputed_data.to_numpy()

            # if opt == 1:
            #     ortho = OApb(ncol)  # OA using Packett Burnman
            #     ortho.to_csv("OA.csv", index=False)
            #     ortho = ortho.to_numpy()
            # else:
            #     if opt == 0:
            #         ortho = np.genfromtxt("OA.csv", delimiter=",")

            # colNames = numerical_cols_for_md

            # runMTS(passdata, faildata, ortho, colNames)

            # --- End of Violin Plot Section ---

            # st.success("Analysis complete! All outliers have been identified.")

            # -----------------------------
            # Step 2: MTS (user-initiated)
            # -----------------------------
            st.write("---")
            st.subheader("Run MTS Feature Importance")

            # Let user pick how to get the orthogonal array
            # oa_source = st.radio(
            #     "Orthogonal Array (OA) source:",
            #     ["Use OA.csv in current folder", "Generate via OApb()"],
            #     index=1,
            #     help="If you pick 'Generate', OApb(ncol) is called to build a Plackett–Burman OA.",
            # )

            # Prepare numeric matrices from the current MTA PASS/FAIL data
            passdata = pass_df[numerical_cols_for_md].copy()
            faildata = fail_df[numerical_cols_for_md].copy()

            for col in passdata.columns:
                passdata[col] = pd.to_numeric(passdata[col], errors="coerce")
            for col in faildata.columns:
                faildata[col] = pd.to_numeric(faildata[col], errors="coerce")

            # Impute from PASS median so shapes/scale match MTA step
            pass_imputed = passdata.fillna(passdata.median()).copy()
            fail_imputed = faildata.fillna(passdata.median()).copy()

            healthy_mat = pass_imputed.to_numpy()  # healthy
            sick_mat = fail_imputed.to_numpy()  # sick
            ncol = healthy_mat.shape[1]
            colNames = numerical_cols_for_md  # for labeling the SNR output

            # --- Button to start MTS ---
            if st.button("Run MTS (Taguchi feature importance)"):
                # Build / load OA
                try:
                    # if oa_source.startswith("Use OA.csv"):
                    #     ortho = np.genfromtxt("OA.csv", delimiter=",")
                    # else:
                    # Generate using your OApb()
                    ortho_df = OApb(ncol)
                    ortho = (
                        ortho_df.to_numpy()
                        if hasattr(ortho_df, "to_numpy")
                        else np.array(ortho_df)
                    )
                    # (Optional) save for reference
                    pd.DataFrame(ortho).to_csv("OA.csv", index=False)

                    with st.spinner("Running MTS (runMTS)..."):
                        runMTS(healthy_mat, sick_mat, ortho, colNames)

                    # Read back the SNR result written by runMTS
                    delta = pd.read_csv("SNRdelta.csv")

                    # Display ranked importance and a chart
                    if {"Test Parameter", "SNR Gain"}.issubset(delta.columns):
                        delta = delta.sort_values(
                            "SNR Gain", ascending=False
                        ).reset_index(drop=True)
                        if "Test Recommendation" not in delta.columns:
                            delta["Test Recommendation"] = np.where(
                                delta["SNR Gain"] < 0.0, "Remove", "Keep"
                            )

                        st.subheader("MTS Feature Importance (SNR Gain)")
                        st.dataframe(delta.head(100))

                        fig = px.bar(
                            delta,
                            x="Test Parameter",
                            y="SNR Gain",
                            color="Test Recommendation",
                            title="Importance vs Test Parameters (MTS / SNR Gain)",
                            hover_data=["Test Parameter", "SNR Gain"],
                        )
                        fig.update_layout(xaxis={"categoryorder": "total descending"})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(
                            "SNRdelta.csv did not have expected columns; showing raw file."
                        )
                        st.dataframe(delta.head(50))

                except Exception as e:
                    st.error(f"MTS failed: {e}")
        else:
            st.warning("MTA Anomaly Score calculation failed.")


if __name__ == "__main__":
    main()
