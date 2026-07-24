from scipy.optimize import OptimizeResult
import numpy as np
import warnings
import time
import torch

def generate_random_problem(m, n):
    A  = torch.randn(m, m + n)
    x  = torch.cat([4 + torch.rand(m), 1 + torch.rand(n)])
    mu = torch.cat([1 + torch.rand(m), 1 + torch.rand(n)])
    lamda = torch.randn(m)
    c = A.T @ lamda + mu
    b = A @ x
    return A, b, c

def convert_to_independent(A, b, tol=1e-12):
    Q, R = torch.linalg.qr(A.T)
    idx = torch.where(torch.abs(R.diagonal()) > tol)[0]
    Q = None
    R = None
    torch.cuda.empty_cache()
    return A.T[:, idx].T.reshape((-1,A.shape[1])), b[idx]

def find_x0(A, b, x0=None, tol=1e-3):
    m, n = A.shape

    if x0 is None:
        x = torch.autograd.Variable(torch.randn(n), requires_grad=True)
    else:
        x = torch.autograd.Variable(x0, requires_grad=True)

    def constraint(x):
        return x.clamp(min=0)

    def constraint_loss_fn(x):
        return (x.clamp(max=0) ** 2).mean()

    def loss_fn(x):
        return ((A @ x - b) ** 2).mean()

    optimizer = torch.optim.Adam([x], lr=1e-5)

    i = 0
    while 1:
        optimizer.zero_grad()
        loss = loss_fn(x) + constraint_loss_fn(x)
        loss.backward()
        optimizer.step()
        x.data = constraint(x.data)
        i += 1
        # print(f'[+] fine tuning x0, i={i} loss={torch.sqrt(loss).item()}')
        if torch.sqrt(loss) < tol:
            break
    return x.data

def linprog(A, b, c, x0=None, eps=1e-5, sigma=0.1, timeout=60, patience=1):
    '''
    minimize cx
       x
    subject to Ax=b
               x>=0
    A.shape = (m, n) = (constraint, variable) = (nrows, ncols)
    '''
    if x0 is None:
        x0 = find_x0(A, b)
    else:
        pass
#         print(A.shape, x0.shape)
#         print(A @ x0)
#         print(b)
#         # Initial solution must be feasible
#         assert(torch.all(x0 >= 0))
#         # Initial solution must be feasible
#         assert(torch.allclose(A @ x0, b, atol=1e-3))
#         exit()

    # Extract dimensions
    m, n = A.shape

    # Reduce to independent constraint only
    A, b = convert_to_independent(A, b)

    # Initialize dual variables
    lambda0 = torch.ones(m)
    mu0     = torch.ones(n)

    # Initialize other things
    success  = False
    nit      = 0
    x_best   = x0
    fun_best = 9999
    err_best = 9999
    n_no_improve = 0

    # Construct of all primal and dual variables
    primal_dual_variables = torch.cat([x0, lambda0, mu0])

    # Define function to unpack all variables
    def unpack(p):
        return p[:n], p[n:n+m], p[n+m:]

    # cache to construct the jacobian matrix
    Z1 = torch.zeros(n, n)
    E1 = torch.eye(n)
    Z2 = torch.zeros([m, m])
    Z3 = torch.zeros([m, n])
    Z4 = Z3.T

    # start time counter
    tic = time.time()

    # Iteratively solve
    while 1:
        # unpack variables
        x, lamda, mu = unpack(primal_dual_variables)
        # measure duality
        duality_measure = x * mu / n
        # right hand side
        rhs = torch.cat([A.T @ lamda + mu - c,
                         A @ x - b,
                         (x * mu) - sigma * duality_measure])
        # compute jacobian
        jacobian = torch.cat((torch.cat((Z1, A.T, E1), dim=1),
                              torch.cat((A, Z2, Z3), dim=1),
                              torch.cat((torch.diag(mu), Z4, torch.diag(x)), dim=1)), dim=0)
        # compute the Jp = -rhs
        try:
            delta_primal_dual_variables = torch.linalg.solve(jacobian, -rhs)
        except torch._C._LinAlgError:
            warnings.warn('torch.linalg.solve error')
            break
        d_x, d_lambda, d_mu = unpack(delta_primal_dual_variables)
        # store best solution
        x, _, _ = unpack(primal_dual_variables)
        # stopping condition
        err = torch.linalg.norm(delta_primal_dual_variables, ord=1) / n
        # print(f'{nit} {fun_best:0.4f} {err} {eps}')
        if err < err_best:
            err_best = err
            x_best   = x
            fun_best = c @ x
            n_no_improve = 0
        else:
            n_no_improve += 1
            if n_no_improve > patience:
                break
        if err < eps:
            success = True
            break
        # stopping condition timeout
        toc = time.time()
        if toc - tic > timeout:
            break
        # step size
        step_size = torch.cat([- x / d_x, -mu / d_mu])
        step_size = step_size[step_size > 0]
        step_size = torch.nan_to_num(step_size, nan=np.inf)
        step_size = min(torch.min(step_size), 1)
        # update
        primal_dual_variables += step_size * delta_primal_dual_variables
        nit += 1

    # return solution
    # x_best[torch.abs(x_best) < eps] = 0
    fun_best = (c @ x_best).item()
    result = OptimizeResult(x=x_best,
                            success=success,
                            nit=nit,
                            fun=fun_best)
    return result

if __name__ == '__main__':
    import time

    m, n = 100, 50
    A, b, c = generate_random_problem(m, n)

    x0 = torch.rand(m + n)
    tic = time.time()
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        result = linprog(A, b, c, x0)
    toc = time.time()
    print(result.x)
    print(m, n, result.success, result.nit, toc - tic)
