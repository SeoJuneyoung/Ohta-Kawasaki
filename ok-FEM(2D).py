from dolfin import *
import numpy as np
import time
import os

# ==========================================
# 출력 폴더 생성
# ==========================================
os.makedirs("output_2d", exist_ok=True)

# ==========================================
# Parameters
# ==========================================
eps = 0.02
sigma = 100.0
nx = ny = 64
dt = 5e-5
steps = 20000
viz_interval = 200 # 200스텝마다 결과 출력 및 파일 저장
m_val = 0.2

# ==========================================
# Mesh & Function Spaces
# ==========================================
mesh = UnitSquareMesh(nx, ny)
dx = Measure("dx", domain=mesh)

P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = MixedElement([P1, P1, P1])

W = FunctionSpace(mesh, ME)
V = FunctionSpace(mesh, "Lagrange", 1)

# ==========================================
# FEM용 부드러운 노이즈 생성 (수치 폭발 방지)
# ==========================================
class SmoothNoise(UserExpression):
    def __init__(self, m_val, **kwargs):
        self.m_val = m_val
        np.random.seed(42)
        # 15개의 무작위 주파수 파동을 중첩하여 부드러운 노이즈 형태 구성
        self.kx = np.random.randint(1, 6, 15)
        self.ky = np.random.randint(1, 6, 15)
        self.amp = (np.random.rand(15) - 0.5) * 0.02
        self.phase = np.random.rand(15) * 2 * np.pi
        super().__init__(**kwargs)

    def eval(self, value, x):
        noise = 0.0
        for i in range(15):
            noise += self.amp[i] * np.sin(self.kx[i]*np.pi*x[0] + self.ky[i]*np.pi*x[1] + self.phase[i])
        value[0] = self.m_val + noise

    def value_shape(self):
        return ()

u0_expr = SmoothNoise(m_val=m_val, degree=1)
u0 = interpolate(u0_expr, V)

# 질량(Mean) 확인
volume = assemble(Constant(1.0)*dx)
m = assemble(u0*dx)/volume
print(f"Target mean: {m_val}, Actual Initial mean: {float(m):.6f}")

# ==========================================
# Initialize Functions
# ==========================================
w = Function(W)
w_n = Function(W)
(u, mu, phi) = split(w)
(u_n, mu_n, phi_n) = split(w_n)

assigner = FunctionAssigner(W.sub(0), V)
assigner.assign(w_n.sub(0), u0)
w.assign(w_n) # 뉴턴 솔버를 위한 초기 추정값 고정

# ==========================================
# Variational Forms
# ==========================================
(v, eta, psi) = TestFunctions(W)
f = u**3 - u_n # Convex-splitting

F1 = (u-u_n)/dt*v*dx + dot(grad(mu), grad(v))*dx
F2 = mu*eta*dx - eps**2*dot(grad(u), grad(eta))*dx - sigma*phi*eta*dx - f*eta*dx
F3 = dot(grad(phi), grad(psi))*dx - (u-Constant(m))*psi*dx

F = F1 + F2 + F3
J = derivative(F, w)

# ==========================================
# phi의 특이성(Singularity)을 제거하기 위한 한 점 고정
# ==========================================
def origin(x, on_boundary):
    # (0, 0) 좌표 근처의 한 점을 식별
    return near(x[0], 0.0, 1e-8) and near(x[1], 0.0, 1e-8)

# W 공간의 세 번째 변수(sub(2), 즉 phi)의 값을 원점에서 0으로 고정
bc_phi = DirichletBC(W.sub(2), Constant(0.0), origin, method="pointwise")
# ==========================================

# [수정된 부분] bcs=[bc_phi] 를 솔버 설정에 추가
problem = NonlinearVariationalProblem(F, w, bcs=[bc_phi], J=J)
solver = NonlinearVariationalSolver(problem)

# ==========================================
# Extract & Save Setup
# ==========================================
# 파라뷰에서 확인할 때 변수 이름을 예쁘게 지정
u_sol = Function(V, name="Phase")
phi_sol = Function(V, name="Potential")

assigner_u = FunctionAssigner(V, W.sub(0))
assigner_phi = FunctionAssigner(V, W.sub(2))

# 파일 시스템에 직접 연결되는 출력 객체 생성
file_u = File("output_2d/u_phase.pvd")
file_phi = File("output_2d/phi_potential.pvd")

def compute_energy(u_func, phi_func):
    grad_term = 0.5*eps**2*assemble(dot(grad(u_func), grad(u_func))*dx)
    bulk = assemble(0.25*(1-u_func**2)**2*dx)
    nonlocal_term = 0.5*sigma*assemble((u_func-m)*phi_func*dx)
    return float(grad_term + bulk + nonlocal_term)

# ==========================================
# Time Loop
# ==========================================
start = time.time()
prev_u = Function(V)

print("Starting simulation...")

# Step 0 (초기 상태) 기록
assigner_u.assign(u_sol, w_n.sub(0))
assigner_phi.assign(phi_sol, w_n.sub(2))
file_u << (u_sol, 0.0)
file_phi << (phi_sol, 0.0)

for n in range(1, steps + 1):
    solver.solve()

    assigner_u.assign(u_sol, w.sub(0))
    assigner_phi.assign(phi_sol, w.sub(2))

    # 주기적으로 에너지 출력 및 파일 기록
    if n % viz_interval == 0:
        E = compute_energy(u_sol, phi_sol)
        diff = np.sqrt(assemble((u_sol-prev_u)**2*dx))
        print(f"Step {n:5d} | Energy = {E:.6e} | L2 diff = {diff:.3e}")
        
        # 중간 스냅샷 저장!
        file_u << (u_sol, float(n * dt))
        file_phi << (phi_sol, float(n * dt))

    prev_u.assign(u_sol)
    w_n.assign(w)

end = time.time()
print(f"Simulation Complete. Elapsed: {end-start:.2f} seconds")
