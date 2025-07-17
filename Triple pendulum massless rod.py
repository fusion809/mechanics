from sympy import symbols, Function, diff, cos, sin, simplify, sqrt, Abs, Eq, solve, latex, Array
from sympy.vector import CoordSys3D
from multiprocessing import Pool, cpu_count
N = CoordSys3D('N');
# Set up symbols
t = symbols('t')
m1 = symbols('m1'); 
m2 = symbols('m2'); 
m3 = symbols('m3'); 
l1 = symbols('11');
l2 = symbols('l2');
l3 = symbols('l3');
g = symbols('g'); 
b1 = symbols('b1'); 
b2 = symbols('b2'); 
b3 = symbols('b3'); 
c1 = symbols("c1"); 
c2 = symbols("c2"); 
c3 = symbols("c3");
theta1=Function('theta1')(t); 
theta2=Function('theta2')(t); 
theta3=Function('theta3')(t); 
x1 = l1*cos(theta1);
y1 = l1*sin(theta1);
r1 = x1*N.i + y1*N.j;
x2 = x1 + l2*cos(theta2);
y2 = y1 + l2*sin(theta2);
r2 = x2*N.i + y2*N.j;
x3 = x2 + l3*cos(theta3);
y3 = y2 + l3*sin(theta3);
r3 = x3*N.i + y3*N.j;
def pythag(x, y):
	return x**2+y**2
dx1 = diff(x1, t);
dy1 = diff(y1, t);
dx2 = diff(x2, t);
dy2 = diff(y2, t);
dx3 = diff(x3, t);
dy3 = diff(y3, t);

def compute_kinetic(m, r):
	T = 0;
	j = 0;
	v = [];
	for i in m:
		v.append(diff(r[j], t))
		T += i*v[j].dot(v[j])/2;
		j += 1
	
	return T
	
def compute_potential(g, m, r):
	V = 0;
	j = 0;
	for i in m:
		y = r[j].dot(N.j)
		V += i * g * y
	
	return V
	
def compute_lagrangian(g, m, r):
	L = compute_kinetic(m, r) - compute_potential(g, m, r)
	return L

def diss_force(b, c, r):
	v = diff(r, t);
	return -(b+c*sqrt(v.dot(v)))*v
	
def gen_diss_force(b, c, r, coord):
	Q = [0 for i in range(len(coord))]
	for j in range(len(coord)):
		for i in range(len(r)):
			v = diff(r[i], t)
			F = diss_force(b[i], c[i], r[i])
			ehat = diff(r[i], coord[j])
			Q[j] += F.dot(ehat)

	return Q
	
def compute_eq_of_motion(g, m, b, c, r):
	t = symbols('t')
	L = compute_lagrangian(g, m, r)
	coord = [theta1, theta2, theta3]
	#Q = gen_diss_force(b, c, r, coord)
	Q = [Function('Qtheta1')(t), Function('Qtheta2')(t), Function('Qtheta3')(t)];
	lhs = [0 for i in range(len(coord))]
	return [Eq(diff(diff(L, diff(coord[i], t)), t) - diff(L, coord[i]), Q[i]) for i in range(len(coord))]
	
m = [m1, m2, m3]
b = [b1, b2, b3]
c = [c1, c2, c3]
r = [r1, r2, r3]
coord = [theta1, theta2, theta3]
eqns = compute_eq_of_motion(g, m, b, c, r)
d2 = [diff(i, (t, 2)) for i in coord];
theta_ddot_syms = symbols('theta1_dd theta2_dd theta3_dd')
#sol = solve(eqn, d2)

# Turn Eq(lhs, rhs) into lhs - rhs == 0
residuals = [eq.lhs - eq.rhs for eq in eqns]

# Substitute second derivatives with dummy symbols (e.g. theta1_dd)
residuals_subs = [res.subs(dict(zip(d2, theta_ddot_syms))) for res in residuals]
from sympy.solvers.solveset import linear_eq_to_matrix

A, b = linear_eq_to_matrix(residuals_subs, theta_ddot_syms)