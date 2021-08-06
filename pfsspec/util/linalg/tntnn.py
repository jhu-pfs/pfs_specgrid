# ===============================================================
# TNT-NN: A Fast Active Set Method for Solving Large Non-Negative 
# Least Squares Problems.
# ===============================================================
# Minimize norm2(b - A * x) with constraints: x(i) >= 0
# ===============================================================
# Authors:	Erich Frahm, frahm@physics.umn.edu
#		    Joseph Myre, myre@stthomas.edu
# Python version: Laszlo Dobos, dobos@jhu.edu
# ===============================================================

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve_triangular

def tntnn(
        A, 
        b, 
        lam = 0, 
        rel_tol = 0, 
        AA = 0, 
        use_AA = False,
        red_c = 0.2, 
        exp_c = 1.2,
        verbose = 0):

        """
        Solves the NNLS problem using the TNT-NN algorithm.

        Minimize norm2(b - A * x) with constraints: x[i] >= 0.

        Arguments:
            A (float): Matrix of size N x M
            b (float): Vector of size N

        Optional keyword arguments:
            lam (float): Tikhonov regularization parameter
            rel_tol (float): Relative tolerance
            AA (float): A^T A matrix to be used as a preconditioner to the LS solver
            use_AA (bool): Use AA for preconditioning
            red_c:
            exp_c:
            verbose: Verbosity level
                0 for no output
                1 for output in a hist matrix file
                2 for output printed to the console
        Returns:
            The solution of the problem, the x vector of size M
        """
        
        tntnn = TntNN(lam, rel_tol, red_c, exp_c, verbose)
        return tntnn.fit(A, b, AA, use_AA)

class TntNN():
    """
    Implements functions to solve the NNLS problem using the TNT-NN algorithm.    
    """

    HIST_VAR = 6

    def __init__(
        self,
        lam = 0, 
        rel_tol = 0, 
        red_c = 0.2, 
        exp_c = 1.2,
        verbose = 0):

        """
        Initializes the TNT-NN object state

        Optional keyword arguments:
            lam (float): Tikhonov regularization parameter
            rel_tol (float): Relative tolerance
            red_c:
            exp_c:
            verbose: Verbosity level
                0 for no output
                1 for output in a hist matrix file
                2 for output printed to the console
        """
        
        self.lam = lam
        self.rel_tol = 0
        self.red_c = 0.2
        self.exp_c = 1.2
        self.verbose = 0
        
        self.save_hist = False
        self.show_hist = False

        self._reset()

    def _reset(self):
        self.x = None
        self.status = None
        self.hist = None

        self.outerloop = 0
        self.totalinnerloops = 0

    def fit(self, A, b, AA=None, use_AA=False):        
        """
        Minimizes norm2(b - A * x) with constraints: x[i] >= 0.

        Optional keyword arguments:
            AA (float): A^T A matrix to be used as a preconditioner to the LS solver
            use_AA (bool): Use AA for preconditioning

        Returns:
            The solution of the problem, the x vector of size M
        """
        
        self._reset()
        
        if self.save_hist:
            self.hist = np.zeros((0, TntNN.HIST_VAR))

        self.x = 0
        self.status = 3                          # unknown failure

        # Get the input matrix size.
        (m, n) = A.shape

        # Check the input vector size.
        if len(b.shape) != 1 or b.shape[0] != m:
            self.status = 2
            raise ValueError('Vector is of wrong size.')

        # ===============================================================
        # Compute A'A one time for use as a preconditioner with the LS 
        # solver.  This is not necessary for all LS solvers.  Unless you 
        # need it for preconditioning it is unlikely you actually need 
        # this step.
        # ===============================================================
        if use_AA:
            if AA.shape != (n, n):
                self.status = 2
                raise ValueError('Matrix is of wrong size.')
        else:
            AA = np.matmul(A.T, A)          # one time

        # ===============================================================
        # AA is a symmetric and positive definite (probably) n x n matrix.
        # If A did not have full rank, then AA is positive semi-definite.
        # Also, if A is very ill-conditioned, then rounding errors can make 
        # AA appear to be indefinite. Modify AA a little to make it more
        # positive definite.
        # ===============================================================
        epsilon = 10 * np.spacing(1) * np.linalg.norm(AA, 1)
        AA = AA + epsilon * np.eye(n)

        # ===============================================================
        # In this routine A will never be changed, but AA might be adjusted
        # with a larger "epsilon" if needed. Working copies called B and BB
        # will be used to perform the computations using the "free" set 
        # of variables.
        # ===============================================================
        
        # ===============================================================
        # Initialize sets as index arrays.
        # ===============================================================
        free_set = np.arange(n, dtype=np.int)
        binding_set = np.zeros((0,), dtype=np.int)
        insertion_set = np.zeros((0,), dtype=np.int)

        # ===============================================================
        # This sets up the unconstrained, core LS solver
        # ===============================================================
        score, x, residual, free_set, binding_set, AA, epsilon, dels, lps, _, _ = \
            self._lsq_solve(A, b, self.lam, AA, epsilon, free_set, binding_set, n)
        
        # ===============================================================
        # Outer Loop.
        # ===============================================================
        insertions = n
        while True:
            self.outerloop += 1
        
            # ===============================================================
            # Save this solution.
            # ===============================================================
            best_score = score
            best_x = x
            best_free_set = free_set.copy()
            best_binding_set = binding_set.copy()
            best_insertions = insertions
            max_insertions = np.int(np.floor(self.exp_c * best_insertions))
            
            # ===============================================================
            # Compute the gradient of the "Normal Equations".
            # ===============================================================
            gradient = np.matmul(A.T, residual)
            
            # ===============================================================
            # Check the gradient components.
            # ===============================================================
            insertion_set = np.where(gradient[binding_set] > 0)[0]
            insertions = insertion_set.size
                    
            # ===============================================================
            # Are we done ?
            # ===============================================================
            if insertions == 0:
                # There were no changes that were feasible. 
                # We are done.
                self.status = 0                      # success 
                if self.save_hist:
                    h = np.array([0, 0, 0, 0, 0, 0])
                    self.hist = np.vstack([self.hist, h])
                    # TODO: save('nnlsq_hist.mat', 'hist');    
                    if self.show_hist:
                        print(self.hist)

                self.x = x
                return self.x
            
            # ===============================================================
            # Sort the possible insertions by their gradients to find the 
            # most attractive variables to insert.
            # ===============================================================
            grad_score = gradient[binding_set[insertion_set]]
            set_index = np.argsort(grad_score)[::-1]
            grad_list = grad_score[set_index]
            insertion_set = insertion_set[set_index]

            # ===============================================================
            # Inner Loop.
            # ===============================================================
            innerloop = 0
            while True:
                innerloop += 1
                self.totalinnerloops += 1

                # ==============================================================
                # Adjust the number of insertions.
                # ==============================================================
                insertions = np.int(np.floor(self.red_c * insertions))
                if insertions == 0:
                    insertions = 1
                if insertions > max_insertions:
                    insertions = max_insertions
                insertion_set = insertion_set[:insertions]

                # ==============================================================
                # Move variables from "binding" to "free".
                # ==============================================================
                free_set = np.hstack([free_set, binding_set[insertion_set]])
                binding_set = np.delete(binding_set, insertion_set)

                # ===============================================================
                # Compute a feasible solution using the unconstrained 
                # least-squares solver of your choice.
                # ===============================================================
                score, x, residual, free_set, binding_set, AA, epsilon, dels, lps, _, _ = \
                    self._lsq_solve(A, b, self.lam, AA, epsilon, free_set, binding_set, insertions)

                # ===============================================================
                # Accumulate history info for algorithm tuning.
                # ===============================================================
                # Each row has 6 values:
                # 1) Outer loop number
                # 2) Inner loop number
                # 3) Total number of inner loops
                # 4) Insertions in this inner loop
                # 5) Deletions required to make the insertions feasible
                # 6) Number of deletion loops required for these insertions
                # ===============================================================
                if self.verbose > 1:
                    print(self.outerloop, innerloop, self.totalinnerloops, insertions, dels, lps)
                if self.save_hist:
                    h = np.array([self.outerloop, innerloop, self.totalinnerloops, insertions, dels, lps])
                    hist = np.vstack([hist, h])
            
                # ===============================================================
                # Check for new best solution.
                # ===============================================================
                if score < best_score * (1 - self.rel_tol):
                    break
                
                # ===============================================================
                # Restore the best solution.
                # ===============================================================
                score = best_score
                x = best_x
                free_set = best_free_set.copy()
                binding_set = best_binding_set.copy()
                max_insertions = np.int(np.floor(exp_c * best_insertions))
                
                # ===============================================================
                # Are we done ?
                # ===============================================================
                if insertions == 1:
                    # The best feasible change did not improve the score. 
                    # We are done.
                    self.status = 0                              # success 
                    if self.verbose > 0:
                        h = np.array([1, 1, 1, 1, 1, 1])
                        self.hist = np.vstack([self.hist, h])
                        # TODO: save('nnlsq_hist.mat', 'hist');
                        if self.show_hist:
                            print(self.hist)
                    return x
            # Inner Loop
        # Outer Loop

        return x

    def _lsq_solve(self, A, b, lam, AA, epsilon, free_set, binding_set, deletions_per_loop):
        """
        Least squares feasible solution using a preconditioned conjugate  
        gradient least-squares solver.
        
        Minimize norm2(b - A * x)
        
        Author: Erich Frahm, frahm@physics.umn.edu
                Joseph Myre, myre@stthomas.edu
        Python version: Laszlo Dobos, dobos@jhu.edu
        """

        # ------------------------------------------------------------
        # Put the lists in order.
        # ------------------------------------------------------------
        free_set = np.sort(free_set)[::-1]
        binding_set = np.sort(binding_set)[::-1]
            
        # ------------------------------------------------------------
        # Reduce A to B.
        # ------------------------------------------------------------
        # B is a matrix that has all of the rows of A, but its
        # columns are a subset of the columns of A. The free_set
        # provides a map from the columns of B to the columns of A.
        B = A[:, free_set]

        # ------------------------------------------------------------
        # Reduce AA to BB.
        # ------------------------------------------------------------
        # BB is a symmetric matrix that has a subset of rows and 
        # columns of AA. The free_set provides a map from the rows
        # and columns of BB to rows and columns of AA.
        BB = AA[free_set, :][:, free_set]

        # ------------------------------------------------------------
        # Adjust with Tikhonov regularization parameter lambda.
        # ------------------------------------------------------------
        if lam > 0:
            B += lam * np.eye(B.shape[0])
            BB += lam * lam * np.eye(BB.shape[0])

        # =============================================================
        # Cholesky decomposition.
        # =============================================================
        while True:
            try:
                R = np.linalg.cholesky(BB).T                 # O(n^3/3)
                break
            except LinAlgError:
                pass

            # It may be necessary to add to the diagonal of B'B to avoid 
            # taking the sqare root of a negative number when there are 
            # rounding errors on a nearly singular matrix. That's still OK 
            # because we just use the Cholesky factor as a preconditioner.
            epsilon *= 10;
            print(epsilon)
            AA = AA + epsilon * np.eye(n)
            BB = AA[free_set, :][:, free_set]
            if lam > 0:
                BB += lam * lam * np.eye(BB.shape[0])
            del R

        # ------------------------------------------------------------
        # Loop until the solution is feasible.
        # ------------------------------------------------------------
        dels = 0
        loops = 0
        lsq_loops = 0
        del_hist = np.zeros((0,), dtype=np.int)
        while True:
            loops += 1
            
            # ------------------------------------------------------------
            # Use PCGNR to find the unconstrained optimum in 
            # the "free" variables.
            # ------------------------------------------------------------
            reduced_x, k = self._pcgnr(B, b, R)
            
            if k > lsq_loops:
                lsq_loops = k
            
            # ------------------------------------------------------------
            # Get a list of variables that must be deleted.
            # ------------------------------------------------------------
            deletion_set = np.where(reduced_x <= 0)[0]
            
            # ------------------------------------------------------------
            # If the current solution is feasible then quit.
            # ------------------------------------------------------------
            if deletion_set.size == 0:
                break
            
            # ------------------------------------------------------------
            # Sort the possible deletions by their reduced_x values to 
            # find the worst violators.
            # ------------------------------------------------------------
            x_score = reduced_x[deletion_set]
            set_index = np.argsort(x_score)
            x_list = x_score[set_index]
            deletion_set = deletion_set[set_index]
            
            # ------------------------------------------------------------
            # Limit the number of deletions per loop.
            # ------------------------------------------------------------
            if deletion_set.size > deletions_per_loop:
                deletion_set = deletion_set[deletions_per_loop:]
            deletion_set = np.sort(deletion_set)[::-1]
            del_hist = np.hstack([del_hist, deletion_set])
            dels += deletion_set.size
            
            # ------------------------------------------------------------
            # Move the variables from "free" to "binding".
            # ------------------------------------------------------------
            binding_set = np.hstack([binding_set, free_set[deletion_set]])
            free_set = np.delete(free_set, deletion_set)
            
            # ------------------------------------------------------------
            # Reduce A to B.
            # ------------------------------------------------------------
            # B is a matrix that has all of the rows of A, but its
            # columns are a subset of the columns of A. The free_set
            # provides a map from the columns of B to the columns of A.
            del B
            B = A[:, free_set]
            
            # ------------------------------------------------------------
            # Reduce AA to BB.
            # ------------------------------------------------------------
            # BB is a symmetric matrix that has a subset of rows and 
            # columns of AA. The free_set provides a map from the rows
            # and columns of BB to rows and columns of AA.
            del BB
            BB = AA[free_set, :][:, free_set]
            
            # ------------------------------------------------------------
            # Adjust with Tikhonov regularization parameter lambda.
            # ------------------------------------------------------------
            if lam > 0:
                B += lam * np.eye(B.shape[0])
                BB += lam * lam * np.eye(BB.shape[0])
            
            # ------------------------------------------------------------
            # Compute R, the Cholesky factor.
            # ------------------------------------------------------------
            R = self._cholesky_delete(R, BB, deletion_set)
            
        # Clear out the B and BB vars to save memory
        #
        del B
        del BB

        # ------------------------------------------------------------
        # Unscramble the column indices to get the full (unreduced) x.
        # ------------------------------------------------------------
        m, n = A.shape
        x = np.zeros((n,), dtype=A.dtype)
        x[free_set] = reduced_x

        # ------------------------------------------------------------
        # Compute the full (unreduced) residual.
        # ------------------------------------------------------------
        residual = b - np.matmul(A, x)

        # ------------------------------------------------------------
        # Compute the norm of the residual.
        # ------------------------------------------------------------
        score = np.sqrt(np.dot(residual, residual))

        return score, x, residual, free_set, binding_set, AA, epsilon, del_hist, dels, loops, lsq_loops

    def _pcgnr(self, A, b, R):
        """
        Iterative Methods for Sparse Linear Systems, Yousef Saad
        Algorithm 9.7 Left-Preconditioned CGNR
        http://www.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf
        
        Author:   Erich Frahm, frahm@physics.umn.edu
                  Joseph Myre, myre@stthomas.edu
        Python version: Laszlo Dobos, dobos@jhu.edu
        """

        m, n = A.shape
        x = np.zeros((n,), dtype=A.dtype)
        r = b
        r_hat = np.matmul(A.T, r)                       # matrix_x_vector, O(mn)
        y = solve_triangular(R.T, r_hat, lower=True)    # back_substitution, O(n^2)
        z = solve_triangular(R, y, lower=False)         # back_substitution, O(n^2)
        # y = np.linalg.solve(R.T, r_hat)
        # z = np.linalg.solve(R, y)
        p = z.copy()
        gamma = np.dot(z, r_hat)
        prev_rr = -1
        for k in range(n):
            w = np.matmul(A, p)                         # matrix_x_vector, O(mn)
            ww = np.dot(w, w)
            if ww == 0:
                return x, k
            alpha = gamma / ww
            x_prev = x
            x += alpha * p;
            r = b - np.matmul(A, x)                 # matrix_x_vector, O(mn)
            r_hat = np.matmul(A.T, r)               # matrix_x_vector, O(mn)
            
            # ---------------------------------------------
            # Enforce continuous improvement in the score.
            # ---------------------------------------------
            rr = np.dot(r_hat, r_hat)
            if (prev_rr >= 0) and (prev_rr <= rr):
                x = x_prev
                return x, k
            prev_rr = rr
            # ---------------------------------------------
            
            y = solve_triangular(R.T, r_hat, lower=True)    # back_substitution, O(n^2)
            z = solve_triangular(R, y, lower=False)         # back_substitution, O(n^2)
            # y = np.linalg.solve(R.T, r_hat)
            # z = np.linalg.solve(R, y)
            gamma_new = np.dot(z, r_hat)
            beta = gamma_new / gamma
            p = z + (beta * p)
            gamma = gamma_new
            if gamma == 0:
                return x, k

        return x, k

    def _cholesky_delete(self, R, BB, deletion_set):
        """
        Compute a new Cholesky factor after deletion of some variables.
    
        Author: Erich Frahm, frahm@physics.umn.edu
        Python version: Laszlo Dobos, dobos@jhu.edu
        """

        m, n = R.shape
        num_deletions = deletion_set.size

        speed_fudge_factor = 0.001;
        if num_deletions > speed_fudge_factor * n:
            # =============================================================
            # Full Cholesky decomposition of BB (on GPUs).
            # =============================================================
            try:
                R = np.linalg.cholesky(BB)                 # O(n^3/3)
            except LinAlgError:
                # This should never happen because we have already added
                # a sufficiently large "epsilon" to AA to do the
                # nonnegativity tests required to create the deleted_set.
                raise        
        else:

            for i in range(num_deletions):
                j = deletion_set[i]
                R = self._qrdel(R, j)

        return R

    def _qrdel(self, R, j):
        """
        This function is just a stripped version of Matlab's qrdelete.
        
        Stolen from:
        http://pmtksupport.googlecode.com/svn/trunk/lars/larsen.m
        """

        R = np.delete(R, j, axis=-1)
        n = R.shape[1]
        for k in range(j, n):
            p = slice(k, k + 2)
            G, x = self._planerot(R[p, k])
            R[p, k] = x                     # Remove extra element in col
            if k < n:
                R[p, k + 1:n + 1] = np.matmul(G, R[p, k + 1:n + 1])     # adjust rest of row
        R = np.delete(R, -1, axis=0)       # remove zero'ed out row
        return R

    def _planerot(self, x):
        """
        Generates a Givens plane rotation.
        """

        if x[1] != 0:
            r = np.linalg.norm(x)
            G = np.array([[x[0], -x[1]], [x[1], x[0]]]) / r
            x = np.array([r, 0])
        else:
            G = np.eye(2, dtype=x.dtype)

        return G, x