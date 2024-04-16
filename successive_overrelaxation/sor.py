import numpy as np

def sor_solver(A, b, omega, initial_guess, convergence_criteria):
    phi = initial_guess[:]
    residual = np.linalg.norm(np.matmul(A, phi) - b)
    while residual > convergence_criteria:
        for i in range(A.shape[0]):
            sigma = 0
            for j in range(A.shape[1]):
                if j != i:
                    sigma += A[i][j] * phi[j]
            phi[i] = (1 - omega) * phi[i] + (omega / A[i][i]) * (b[i] - sigma)
        residual = np.linalg.norm(np.matmul(A, phi) - b)
    return phi, residual

def main():
    residual_convergence = 1e-15
   
    arr = [[ 40, -16,   0, -16,   0,   0],
           [-16,  97, -36,   0, -36,   0],
           [  0, -36, 180,   0,   0, -64],
           [-16,   0,   0,  97, -36,   0],
           [  0, -36,   0, -36, 234, -81],
           [  0,   0, -64,   0, -81, 433]]

    A = np.asarray(arr)
    b = np.ones(6)

    U = np.triu(A, 1)
    L = np.tril(A, -1)
    D = np.tril(np.triu(A))
    rho = max(np.linalg.eigvals(np.matmul(np.linalg.inv(D), L + U)), key=abs)
    omega = 2 / (1 + (1 - rho * rho) ** 0.5)

    initial_guess = np.zeros(6)

    phi, res = sor_solver(A, b, omega, initial_guess, residual_convergence)
    print("Решение:")
    for i in phi:
        print(round(i, 6))
    print()

    print("Норма невязки = {:.2e}\n".format(res))

    print("Решение, полученное решателем:")
    for i in np.linalg.solve(np.array(arr), b):
        print(round(i, 6))

main()