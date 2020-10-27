from __future__ import division
from sys import stdout
from sklearn.metrics import pairwise_kernels
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.stats import gamma


def MMD2u(K, m, n):
    """
    The MMD^2_u unbiased statistic.

    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
           1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
           2.0 / (m * n) * Kxy.sum()


def compute_null_distribution(K, m, n, iterations=1000, verbose=False,
                              random_state=None, marker_interval=100):
    """
    Compute the bootstrap null-distribution of MMD2u.

    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)

    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i),
            stdout.flush()
        idx = rng.permutation(m + n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    if verbose:
        print("")

    return mmd2u_null


def kernel_two_sample_test(X, Y, kernel_function='rbf', iterations=1000, marker_interval=100,
                           verbose=False, random_state=None, **kwargs):
    """
    Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.

    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    m = len(X)
    n = len(Y)
    XY = np.vstack([X, Y])
    K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
    mmd2u = MMD2u(K, m, n)
    if verbose:
        print("MMD^2_u = %s" % mmd2u)
        print("Computing the null distribution.")

    mmd2u_null = compute_null_distribution(K, m, n, iterations=iterations, verbose=verbose,
                                           random_state=random_state, marker_interval=marker_interval)
    p_value = max(1.0 / iterations, (mmd2u_null > mmd2u).sum() /
                  float(iterations))
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0 / iterations))

    return mmd2u, mmd2u_null, p_value


def perturb_specific_gene(anndata, sgRNA_list, perturb="all", gene_name='Mrpl15',
                          max_choice=1000, kernel_function='rbf',
                          iterations=1000, marker_interval=100, ks_compare=True,
                          verbose=True, plot=True):
    """
    Calculate whether the gene is influenced by the perturbation using kernel two-sample test

    Parameters
    ----------
    perturb : the name of perturbation
    gene_name : the name of the target gene
    max_choice : the maximum sample number of both the perturbed and unperturbed group
    kernel_function : the kernel_function of kernel two-sample test
    iteration : iteration of the null distribution
    marker_interval : the intervel of report
    verbose : (bool) whether to report the process
    plot : (bool) whether to plot


    Return
    -------
    mmd2u : the mmd matrix
    mmd2u_null : the null distribution
    p_value : the p_value of the kernel two-sample test

    """

    location = list(anndata.var['gene_ids'].index).index(gene_name)

    if perturb == "all":
        booll = (anndata.obs.iloc[:, 0:len(sgRNA_list)].sum(axis=1).values > 0) * 1
    else:
        booll = anndata.obs[perturb].values

    X = booll * anndata.X[:, location]
    Y = - (booll - 1) * anndata.X[:, location]

    X = X[X != 0]
    Y = Y[Y != 0]
    choice = min(max_choice, len(X), len(Y))

    X = np.random.choice(X, choice, replace=False)
    Y = np.random.choice(Y, choice, replace=False)

    if ks_compare:
        print(X, Y)
        print(stats.ks_2samp(X, Y))

    X = [[el] for el in X]
    Y = [[el] for el in Y]

    if verbose:
        print('The samples of both perturbed and unperturbed gene expressions are ready.')

    sigma2 = np.median(pairwise_distances(X, Y, metric='euclidean')) ** 2
    mmd2u, mmd2u_null, p_value = kernel_two_sample_test(X, Y, kernel_function,
                                                        iterations=iterations,
                                                        marker_interval=marker_interval,
                                                        gamma=1.0 / sigma2, verbose=True)

    if plot:
        plt.figure()
        prob, bins, patches = plt.hist(mmd2u_null, bins=50, normed=True)
        plt.plot(mmd2u, prob.max() / 30, 'w*', markersize=24, markeredgecolor='k',
                 markeredgewidth=2, label="$MMD^2_u = %s$" % mmd2u)
        plt.xlabel('$MMD^2_u$')
        plt.ylabel('$p(MMD^2_u)$')
        plt.legend(numpoints=1)
        plt.title('$MMD^2_u$: null-distribution and observed value. $p$-value=%s'
                  % p_value)

    return mmd2u, mmd2u_null, p_value


def rbf_dot(pattern1, pattern2, deg):
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = np.sum(pattern1 * pattern1, 1).reshape(size1[0], 1)
    H = np.sum(pattern2 * pattern2, 1).reshape(size2[0], 1)

    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))

    H = Q + R - 2 * np.dot(pattern1, pattern2.T)
    H = np.exp(-H / 2 / (deg ** 2))

    return H


def hsic_gam(X, Y, alph=0.05):
    """

    X, Y are numpy vectors with row - sample, col - dim
    alph is the significance level
    auto choose median to be the kernel width

    """

    n = X.shape[0]

    # ----- width of X -----
    Xmed = X

    G = np.sum(Xmed * Xmed, 1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * np.dot(Xmed, Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_x = np.sqrt(0.5 * np.median(dists[dists > 0]))
    # ----- -----

    # ----- width of X -----
    Ymed = Y

    G = np.sum(Ymed * Ymed, 1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * np.dot(Ymed, Ymed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_y = np.sqrt(0.5 * np.median(dists[dists > 0]))
    # ----- -----

    bone = np.ones((n, 1), dtype=float)
    H = np.identity(n) - np.ones((n, n), dtype=float) / n

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)

    Kc = np.dot(np.dot(H, K), H)
    Lc = np.dot(np.dot(H, L), H)

    testStat = np.sum(Kc.T * Lc) / n

    varHSIC = (Kc * Lc / 6) ** 2

    varHSIC = (np.sum(varHSIC) - np.trace(varHSIC)) / n / (n - 1)

    varHSIC = varHSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))

    muX = np.dot(np.dot(bone.T, K), bone) / n / (n - 1)
    muY = np.dot(np.dot(bone.T, L), bone) / n / (n - 1)

    mHSIC = (1 + muX * muY - muX - muY) / n

    al = mHSIC ** 2 / varHSIC
    bet = varHSIC * n / mHSIC

    thresh = gamma.ppf(1 - alph, al, scale=bet)[0][0]

    return (testStat, thresh)


def independence_between_genes(anndata, gene_name1="Mrpl15", gene_name2="Lypla1", perturb='Nfkb1',
                               verbose=True):
    location1 = list(anndata.var['gene_ids'].index).index(gene_name1)
    location2 = list(anndata.var['gene_ids'].index).index(gene_name2)

    booll = anndata.obs[perturb].values

    X = booll.reshape(anndata.n_obs, 1) * anndata.X[:, [location1, location2]]
    X = X[~np.all(X == 0, axis=1)]

    X1 = np.asarray([[el] for el in X[:, 0]])
    X2 = np.asarray([[el] for el in X[:, 1]])

    result = hsic_gam(X1, X2, alph=0.05)
    if verbose:
        print("testStat : ", result[0], "; thresh : ", result[1])

    return result[0] > result[1]
