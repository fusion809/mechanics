from sympy import symbols, Function, diff, cos, sin, simplify, sqrt, Abs, Eq, solve, latex, integrate, factor
from sympy.vector import CoordSys3D
from multiprocessing import Pool, cpu_count
from math import floor
N = CoordSys3D('N');
# Set up symbols
t = symbols('t')
m1b = symbols('m1b'); 
m2b = symbols('m2b'); 
m1r = symbols('m1r'); 
m2r = symbols('m2r'); 
l1 = symbols('11');
l2 = symbols('l2');
k1 = symbols('k1'); 
k2 = symbols('k2'); 
g = symbols('g'); 
b1r = symbols('b1r'); 
b2r = symbols('b2r'); 
b1b = symbols('b1b'); 
b2b = symbols('b2b'); 
c1r = symbols("c1r"); 
c1b = symbols("c1b"); 
c2r = symbols("c2r"); 
c2b = symbols("c2b");
M1 = symbols("M1");
M2 = symbols("M2");
mu1 = symbols("mu1");
mu2 = symbols("mu2");

# Functions
r1 = Function('r1')(t); 
dr1 = diff(r1, t);
theta1 = Function('theta1')(t); 
dtheta1 = diff(theta1, t);
r2 = Function('r2')(t); 
dr2 = diff(r2, t);
theta2 = Function('theta2')(t);
dtheta2 = diff(theta2, t);
coords = [r1, r2, theta1, theta2]
dcoords = [dr1, dr2, dtheta1, dtheta2]
M1 = symbols("M1");
M2 = symbols("M2");
mu1 = symbols("mu1");
mu2 = symbols("mu2");

def Lagrangian(m, g, k, l, coords, dcoords):
	M1 = m[0];
	M2 = m[1];
	mu1 = m[2];
	mu2 = m[3];
	k1 = k[0];
	k2 = k[1];
	l1 = l[0];
	l2 = l[1];
	r1 = coords[0];
	r2 = coords[1];
	theta1 = coords[2];
	theta2 = coords[3];
	dr1 = dcoords[0];
	dr2 = dcoords[1];
	dtheta1 = dcoords[2];
	dtheta2 = dcoords[3];
	Delta = theta2-theta1;
	L = M1/2 * (dr1**2 + r1**2 * dtheta1**2) + M2/2*(dr2**2 + r2**2*dtheta2**2) + mu2*(cos(Delta)*(dr1*dr2 + r1*r2*dtheta1*dtheta2)+sin(Delta)*(r1*dr2*dtheta1 - dr1*r2*dtheta2)-g*r2*sin(theta2)) - mu1*g*r1*sin(theta1) - (k1*(r1-l1)**2+k2*(r2-l2)**2)/2;
	return L

m = [M1, M2, mu1, mu2];
l = [l1, l2];
k = [k1, k2];
dr1 = diff(r1, t)
dr2 = diff(r2, t)
dtheta1 = diff(theta1, t)
dtheta2 = diff(theta2,t)

def ELE(m, g, k, l, coords, dcoords):
	L = Lagrangian(m, g, k, l, coords, dcoords)
	Q = [Function('Qr1')(t), Function('Qr2')(t), Function('Qtheta1')(t), Function('Qtheta2')(t)]
	return [Eq(simplify(diff(diff(L, diff(coords[i], t)), t) - diff(L, coords[i])), Q[i]) for i in range(len(coords))]

coords_ddot_syms = symbols('r1_dd r2_dd theta1_dd theta2_dd')
d2 = [diff(i, (t, 2)) for i in coords];
equations = ELE(m, g, k, l, coords, dcoords)
# Turn Eq(lhs, rhs) into lhs - rhs == 0
residuals = [eq.lhs - eq.rhs for eq in equations]

# Substitute second derivatives with dummy symbols (e.g. theta1_dd)
residuals_subs = [res.subs(dict(zip(d2, coords_ddot_syms))) for res in residuals]
from sympy.solvers.solveset import linear_eq_to_matrix

A, RHS = linear_eq_to_matrix(residuals_subs, coords_ddot_syms)

M1, M2, mu1, mu2, Delta = symbols("M1, M2, mu1, mu2, Delta")
subs_dict = {
	theta1-theta2: -Delta,
	theta2-theta1: Delta,
	m1b + m1r/3 + m2b + m2r: M1,
	(3*m1b + m1r + 3*m2b + 3*m2r)/3: M1,
	3*m1b + m1r + 3*m2b + 3*m2r: 3*M1,
	m1b + m1r/2 + m2b + m2r: mu1,
	(2*m1b+m1r+2*m2b+2*m2r)/2: mu1,
	2*m1b+m1r+2*m2b+2*m2r: 2*mu1,
	m2b + m2r/3: M2,
	(3*m2b+m2r)/3: M2,
	3*m2b+m2r: 3*M2,
	m2b + m2r/2: mu2,
	(2*m2b + m2r)/2: mu2,
	(2*m2b + m2r): 2*mu2
}

A_simplified = A.applyfunc(lambda x: simplify(x).subs(subs_dict, simultaneous=True))
RHS_simplified = RHS.applyfunc(lambda x: simplify(x).subs(subs_dict, simultaneous=True))

from sympy import shape
for i in range(shape(A_simplified)[0]):
	print("i = " + str(i) + ", A[i,:] = ")
	print(A_simplified[i,:].subs(subs_dict, simultaneous=True))
	print("b[i] = ")
	print(RHS_simplified[i].subs(subs_dict, simultaneous=True))