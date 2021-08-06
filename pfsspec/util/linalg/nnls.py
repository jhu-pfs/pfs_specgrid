import numpy as np

def svd_solve(A, b, need_cov=False):
    U, s, Vh = np.linalg.svd(A)

    # solve A * x = b
    S = np.zeros((Vh.shape[0], U.shape[0]), dtype=s.dtype)
    np.fill_diagonal(S, 1.0 / s)
    x = np.matmul(np.matmul(Vh.T, S), np.dot(U.T, b))

    # covariance matrix
    if need_cov:
        C = np.matmul(Vh.T, Vh)
        return x, C
    else:
        return x

# Non-negative least squares from Lawson & Hanson (1974)
def nnls(A, b, eps=0.0, maxiter=None):
    maxiter = maxiter or 3 * A.shape[-1]

    # solution vector
    x = np.zeros(A.shape[1], dtype=A.dtype)

    # indexes for filteres argmax
    idx = np.arange(x.shape[0])

    # Z contains indexes held at zero, if true, element is in Z
    # all elements are in Z, P is Z's complementary set!
    Z = np.array(A.shape[1] * [True])

    # outer loop
    l1count = 0
    while True:       
        # negative gradient vector
        w = np.matmul(A.T, b - np.matmul(A, x))

        # if Z is empty or all elements of w with indices in Z are <= 0 we have a solution
        empty = np.all(~Z)
        allwsmall = np.max(w[Z]) <= eps
        if empty or allwsmall:
            return x

        # wmaxidx points to maximum element in w, given its in Z
        # it is positive, move its index from Z to P
        wmaxidx = idx[Z][np.argmax(w[Z])]
        Z[wmaxidx] = False;		# Z -> P since they're complementary sets

        # inner loop
        while True:         
            # construct matrix from columns of A which are not in Z
            Apos = A[:, ~Z]

            # solve unconstrained least square Apos * z = b;
            xpos = svd_solve(Apos, b)

            # set all elements of z to 0, whose index is in Z
            # check whether all other elements of z are >= 0,
            # if it is true, z is a good solution, set x = z and countinue to main loop
            z = np.zeros_like(x)
            z[~Z] = xpos
            
            # if all positive this is the solution
            if np.min(xpos) > 0.0:
                x = z
                break

            # find the minimum of alpha for elements not in Z and where z <= 0
            m = ~Z & (z <= 0)
            alpha = np.min(x[m] / (x[m] - z[m]))

            x = x + (alpha * (z - x))

            # move all indices not already in Z to Z where x <= 0.0
            Z[~Z] |= (x[~Z] <= 0.0)

        # end inner loop
    # end outer loop