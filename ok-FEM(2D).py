from dolfin import *
import numpy as np
import time

# -----------------------------
# Parameters
# -----------------------------
eps = 0.02
sigma = 100.0

nx = ny = 64

# 초반의 급격한 상분리를 견디기 위해 낮춘 dt 유지
dt = 5e-5 
steps = 20000
viz_interval = 1000 

# 질량 비율
m_val = 0.2 

# -----------------------------
# Mesh
# -----------------------------
mesh = UnitSquareMesh(nx, ny)
dx = Measure("dx", domain=mesh)

# -----------------------------
# Function spaces
# -----------------------------
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = MixedElement([P1, P1, P1])

W = FunctionSpace(mesh, ME)
V = FunctionSpace(mesh, "Lagrange", 1)

# -----------------------------
# Initial condition (Spinodal Decomposition)
# -----------------------------
# 재현성을 위한 시드 고정 및 무작위 노이즈 부여
np.random.seed(42)
u0 = Function(V)

coords = V.tabulate_dof_coordinates().reshape((-1, 2))

# m_val(0.2)을 중심으로 진폭 0.05의 무작위 노이즈(-0.025 ~ +0.025) 생성
vals = m_val + 0.05 * (np.random.rand(len(coords)) - 0.5)

u0.vector()[:] = vals

# 정확한 질량 보존을 위해 FEniCS 내부적으로 한 번 더 적분하여 m 계산
volume = assemble(Constant(1.0)*dx)
m = assemble(u0*dx)/volume

print(f"Target mean: {m_val}, Actual Initial mean: {float(m):.6f}")

# -----------------------------
# Mixed functions
# -----------------------------
w = Function(W)
w_n = Function(W)

(u, mu, phi) = split(w)
(u_n, mu_n, phi_n) = split(w_n)

# initialize
assigner = FunctionAssigner(W.sub(0), V)
assigner.assign(w_n.sub(0), u0)

# 뉴턴 솔버 초기 추정값 유지
w.assign(w_n)

# -----------------------------
# Test functions
# -----------------------------
(v, eta, psi) = TestFunctions(W)

# -----------------------------
# Nonlinear function
# -----------------------------
# Convex-Splitting 유지
f = u**3 - u_n

# -----------------------------
# Variational forms
# -----------------------------
F1 = (u-u_n)/dt*v*dx + dot(grad(mu), grad(v))*dx

F2 = mu*eta*dx - eps**2*dot(grad(u), grad(eta))*dx \
     - sigma*phi*eta*dx \
     - f*eta*dx

F3 = dot(grad(phi), grad(psi))*dx \
     - (u-Constant(m))*psi*dx

F = F1 + F2 + F3

# Jacobian
J = derivative(F, w)

# nonlinear solver
problem = NonlinearVariationalProblem(F, w, J=J)
solver = NonlinearVariationalSolver(problem)

# -----------------------------
# Extract spaces
# -----------------------------
u_sol = Function(V)
phi_sol = Function(V)

assigner_u = FunctionAssigner(V, W.sub(0))
assigner_phi = FunctionAssigner(V, W.sub(2))

def compute_energy(u_func, phi_func):
    grad_term = 0.5*eps**2*assemble(dot(grad(u_func), grad(u_func))*dx)
    bulk = assemble(0.25*(1-u_func**2)**2*dx)
    nonlocal_term = 0.5*sigma*assemble((u_func-m)*phi_func*dx)
    return float(grad_term + bulk + nonlocal_term)

# -----------------------------
# Time loop
# -----------------------------
start = time.time()
prev_u = Function(V)

for n in range(steps):
    solver.solve()

    assigner_u.assign(u_sol, w.sub(0))
    assigner_phi.assign(phi_sol, w.sub(2))

    if n % viz_interval == 0 or n < 10:
        E = compute_energy(u_sol, phi_sol)

        if n == 0:
            diff = np.sqrt(assemble((u_sol-u0)**2*dx))
        else:
            diff = np.sqrt(assemble((u_sol-prev_u)**2*dx))

        print(f"step {n:6d}  E={E:.6e}  L2diff={diff:.3e}")

    prev_u.assign(u_sol)
    w_n.assign(w)

end = time.time()
print("Elapsed:", end-start)

# -----------------------------
# Save
# -----------------------------
File("u_final_noise.pvd") << u_sol
File("phi_final_noise.pvd") << phi_sol