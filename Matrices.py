import numpy as np
from scipy.sparse import eye, diags, block_diag, block_array
import scipy


'''Basic Simulation Matrices'''
def tau_calc(N_k, beta=True):
    '''
    Calculate the tau matrix.

    Parameters:
    N_k (int): Number of blocks in the y-direction.
    beta (bool): Flag for periodic boundary conditions in the y-direction. Default is 1.

    Returns:
    scipy.sparse.dia_matrix: The tau matrix.
    '''
    # Create a diagonal sparse matrix with ones on the main diagonal, and add ones to the first superdiagonal
    # If beta is set, add ones to the (1-N_k)th diagonal (for periodic boundary conditions)
    tau = diags([[1], [1], [beta*1]], [0, 1, 1-N_k], (N_k,N_k))
    return tau
def iota_calc(N_k, beta=True):
    '''
    Calculate the iota matrix.

    Parameters:
    N_k (int): Number of blocks in the y-direction.
    beta (bool): Flag for periodic boundary conditions in the y-direction. Default is 1.

    Returns:
    scipy.sparse.dia_matrix: The iota matrix.
    '''
    # Create a diagonal sparse matrix with ones on the main diagonal, and subtract ones from the first superdiagonal
    # If beta is set, subtract ones from the (1-N_k)th diagonal (for periodic boundary conditions)
    iota = diags([[1], [-1], [-beta*1]], [0, 1, 1-N_k], (N_k,N_k))
    return iota

def T_calc(N_j, N_k, alpha=True, beta=True):
    '''
    Calculate the T matrix using the tau matrix.

    Parameters:
    N_j (int): Number of blocks in the x-direction.
    N_k (int): Number of blocks in the y-direction.
    alpha (bool): Flag for periodic boundary conditions in the x-direction. Default is 1.
    beta (bool): Flag for periodic boundary conditions in the y-direction. Default is 1.

    Returns:
    scipy.sparse.block_matrix: The T matrix.
    '''
    # Calculate the tau matrix using the tau_calc function
    tau = tau_calc(N_k, beta)
    # Create a block matrix with tau matrices along the diagonal and first superdiagonal
    T = block_array([[None]*i+[tau, tau]+[None]*(N_j-2-i) for i in range(N_j-1)]+[[None]*(N_j-1)+[tau]])
    # If alpha is set, add a block structure to T to handle periodic boundary conditions along the x-direction
    if alpha:
        T += block_array([[None, diags([0], 0, (N_k*N_j-N_k,N_k*N_j-N_k))], [tau, None]])
    return T
def X_calc(N_j, N_k, alpha=True, beta=True):
    '''
    Calculate the X matrix using the tau matrix.

    Parameters:
    N_j (int): Number of blocks in the x-direction.
    N_k (int): Number of blocks in the y-direction.
    alpha (bool): Flag for periodic boundary conditions in the x-direction. Default is 1.
    beta (bool): Flag for periodic boundary conditions in the y-direction. Default is 1.

    Returns:
    scipy.sparse.block_matrix: The X matrix.
    '''
    # Calculate the tau matrix using the tau_calc function
    tau = tau_calc(N_k, beta)
    # Create a block matrix with tau matrices along the diagonal and -tau matrices along the first superdiagonal
    X = block_array([[None]*i+[tau, -tau]+[None]*(N_j-2-i) for i in range(N_j-1)]+[[None]*(N_j-1)+[tau]])
    # If alpha is set, add a block structure to T to handle periodic boundary conditions along the x-direction
    if alpha:
        X -= block_array([[None, diags([0], 0, (N_k*N_j-N_k,N_k*N_j-N_k))], [-tau, None]])
    return X
def Y_calc(N_j, N_k, alpha=True, beta=True):
    '''
    Calculate the Y matrix using the iota matrix.

    Parameters:
    N_j (int): Number of blocks in the x-direction.
    N_k (int): Number of blocks in the y-direction.
    alpha (bool): Flag for periodic boundary conditions in the x-direction. Default is 1.
    beta (bool): Flag for periodic boundary conditions in the y-direction. Default is 1.

    Returns:
    scipy.sparse.block_matrix: The Y matrix.
    '''
    # Calculate the iota matrix using the iota_calc function
    iota = iota_calc(N_k, beta)
    # Create a block matrix with iota matrices along the diagonal and the first superdiagonal
    Y = block_array([[None]*i+[iota, iota]+[None]*(N_j-2-i) for i in range(N_j-1)]+[[None]*(N_j-1)+[iota]])
    # If alpha is set, add a block structure to Y to handle periodic boundary conditions along the x-direction
    if alpha:
        Y += block_array([[None, diags([0], 0, (N_k*N_j-N_k,N_k*N_j-N_k))], [iota, None]])
    return Y

'''Full Simulation Matrices'''
def M_calc(N_j, N_k, alpha=True, beta=True, V=0, hbar=1, c=1, m=1, dx=1, dy=1, dt=1/100):
    '''
    Calculate the M matrix for the simulation.

    Parameters:
    N_j (int): Number of blocks in the x-direction.
    N_k (int): Number of blocks in the y-direction.
    alpha (bool): Flag for periodic boundary conditions in the x-direction. Default is 1.
    beta (bool): Flag for periodic boundary conditions in the y-direction. Default is 1.
    V (float or array-like): Potential. Default is 0.
    hbar (float): Reduced Planck constant. Default is 1.
    c (float): Speed of light. Default is 1.
    m (float): Mass parameter. Default is 1.
    dx (float): Grid spacing in the x-direction. Default is 1.
    dy (float): Grid spacing in the y-direction. Default is 1.
    dt (float): Time step. Default is 1/100.

    Returns:
    scipy.sparse.block_matrix: The M matrix.
    '''
    # Calculate T, X, and Y using the respective functions
    T = T_calc(N_j, N_k, alpha, beta)
    X = X_calc(N_j, N_k, alpha, beta)
    Y = Y_calc(N_j, N_k, alpha, beta)
    # Make sure V is the right size
    if not np.isscalar(V):
        V = np.asarray(V)
        expected_size = N_j * N_k
        if V.size > expected_size:
            V = V[:expected_size]
        elif V.size < expected_size:
            V = np.pad(V, (0, expected_size - V.size), 'constant')
        V = diags(V)
    else:
        V = diags([V], shape = (N_j * N_k, N_j * N_k))
    # Calculate the M matrix using the provided parameters and precomputed matrices
    M = block_diag((T, T))/dt
    M += 1j/(2*hbar)*block_diag((V*T, V*T))
    M += 1j*m*c*c/(2*hbar)*block_diag((T, -T))
    M -= c/dx*block_array([[None, X],[X, None]])
    M += 1j*c/dy*block_array([[None, Y],[-Y, None]])
    return M
def P_calc(N_j, N_k, alpha=True, beta=True, V=0, hbar=1, c=1, m=1, dx=1, dy=1, dt=1/100):
    '''
    Calculate the P matrix for the simulation.

    Parameters:
    N_j (int): Number of blocks in the x-direction.
    N_k (int): Number of blocks in the y-direction.
    alpha (bool): Flag for periodic boundary conditions in the x-direction. Default is 1.
    beta (bool): Flag for periodic boundary conditions in the y-direction. Default is 1.
    V (float or array-like): Potential parameter. Default is 0.
    hbar (float): Reduced Planck constant. Default is 1.
    c (float): Speed of light. Default is 1.
    m (float): Mass parameter. Default is 1.
    dx (float): Grid spacing in the x-direction. Default is 1.
    dy (float): Grid spacing in the y-direction. Default is 1.
    dt (float): Time step. Default is 1/100.

    Returns:
    scipy.sparse.block_matrix: The P matrix.
    '''
    # Calculate T, X, and Y using the respective functions
    T = T_calc(N_j, N_k, alpha, beta)
    X = X_calc(N_j, N_k, alpha, beta)
    Y = Y_calc(N_j, N_k, alpha, beta)
    # Make sure V is the right size
    if not np.isscalar(V):
        V = np.asarray(V)
        expected_size = N_j * N_k
        if V.size > expected_size:
            V = V[:expected_size]
        elif V.size < expected_size:
            V = np.pad(V, (0, expected_size - V.size), 'constant')
        V = diags(V)
    # Calculate the P matrix using the provided parameters and precomputed matrices
    P = block_diag((T, T))/dt
    P -= 1j/(2*hbar)*block_diag((V*T, V*T))
    P -= 1j*m*c*c/(2*hbar)*block_diag((T, -T))
    P += c/dx*block_array([[None, X],[X, None]])
    P -= 1j*c/dy*block_array([[None, Y],[-Y, None]])
    return P


if __name__ == "__main__":
    '''Simulation parameters'''
    N_j = 3
    N_k = 3
    alpha = 1       # Periodic along x
    beta = 1        # Periodic along y
    '''Basic Matrices'''
    T = T_calc(N_j, N_k, alpha, beta)
    X = X_calc(N_j, N_k, alpha, beta)
    Y = Y_calc(N_j, N_k, alpha, beta)
    '''Simulation Matrices'''
    M = M_calc(N_j, N_k, alpha, beta)
    P = P_calc(N_j, N_k, alpha, beta)