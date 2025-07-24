from sympy import symbols, Function, diff, cos, sin, simplify, sqrt, Abs, Eq, solve, latex, integrate, factor, collect, shape
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
theta1=Function('theta1')(t); 
r2 = Function('r2')(t); 
theta2=Function('theta2')(t);

# Bob 1
x1b = r1*cos(theta1);
y1b = r1*sin(theta1);
y1r = r1*sin(theta1)/2;
r1b = x1b * N.i + y1b * N.j;
v1b = diff(r1b, t);
v1b_sq = v1b.dot(v1b);
v1b_mag = sqrt(v1b_sq);
e1b_r1 = diff(r1b, r1);
e1b_r2 = diff(r1b, r2);
e1b_th1 = diff(r1b, theta1);
e1b_th2 = diff(r1b, theta2);

# Bob 2
x2b = x1b+r2*cos(theta2);
y2b = y1b + r2*sin(theta2);
y2r = y1b + r2*sin(theta2)/2;
r2b = x2b * N.i + y2b * N.j;
v2b = diff(r2b, t);
v2b_sq = v2b.dot(v2b);
v2b_mag = sqrt(v2b_sq);
e2b_r1 = diff(r2b, r1);
e2b_r2 = diff(r2b, r2);
e2b_th1 = diff(r2b, theta1);
e2b_th2 = diff(r2b, theta2);

# Rod 1
x1r = x1b/2;
y1r = y1b/2;
r1r = x1r*N.i+y1r*N.j;
v1r = diff(r1r, t);
# Rod 2
x2r = x1b + r2*cos(theta2)/2;
y2r = y1b + r2*sin(theta2)/2;
r2r = x2r*N.i+y2r*N.j;
v2r = diff(r2r, t);


def diss_force(b, c, pos):
	v = diff(pos, t);
	F = -(b+c*sqrt(v.dot(v)))*v;
	return F;

def gen_diss_force(b, c, l, pos, coords):
	Q = [0 for i in range(len(coords))];
	for i in range(len(coords)):
		for j in range(len(b)):
			F = diss_force(b[j], c[j], pos[j])
			ehat = diff(pos[j], coords[i]);
			Q[i] += F.dot(ehat);
		Q[i] = simplify(Q[i])
	return Q

def compute_kinetic(m, pos):
	T = 0;
	v = [diff(posi, t) for posi in pos];
	for i in range(len(l)):
		# Translational energy for rods, 2*i and 2*(i-1) corresponds to the bob velocities
		if (i == 0):
			T += m[2*i+1]/6 * v[2*i].dot(v[2*i])
		else:
			T += m[2*i+1]/6 * (v[2*(i-1)].dot(v[2*(i-1)])+v[2*i].dot(v[2*(i-1)])+v[2*i].dot(v[2*i]))
		# Kinetic energy of bobs
		T += m[2*i]/2 * v[2*i].dot(v[2*i]);
	
	return T

def compute_potential(g, k, m, l, r, y):
	V = 0;
	for j in range(len(y)):
		V += m[j] * g * y[j]
	
	for i in range(len(k)):
		V += k[i]/2 * (r[i]-l[i])**2;
		
	return V
	
	
def compute_equations_motion(g, k, m, l, coords, y, poscent):
	r1 = coords[0];
	dr1 = diff(r1,t);
	r2 = coords[1];
	dr2 = diff(r2, t)
	theta1 = coords[2];
	dtheta1 = diff(theta1, t)
	theta2 = coords[3];
	dtheta2 = diff(theta2, t)
	Delta = theta2 - theta1;
	L = M1/2*(dr1**2+r1**2*dtheta1**2) + M2/2*(dr2**2+r2**2*dtheta2**2) + mu2*(cos(Delta)*(dr1*dr2 + r1*r2*dtheta1*dtheta2) + sin(Delta)*(r1*dr2*dtheta1 - dr1*r2*dtheta2)-g*r2*sin(theta2)) - mu1*g*r1*sin(theta1) - k1*(r1-l1)**2/2 - k2*(r2-l2)**2/2;
	Q = [Function('Qr1')(t), Function('Qr2')(t), Function('Qtheta1')(t), Function('Qtheta2')(t)]
	return [Eq(diff(diff(L, diff(coords[i], t)), t) - diff(L, coords[i]), Q[i]) for i in range(len(coords))]
	
y = [y1b, y1r, y2b, y2r];
r = [r1, r2];
m = [m1b, m1r, m2b, m2r];
l = [l1, l2];
k = [k1, k2];
theta = [theta1, theta2];
coords = r + theta;

d2 = [diff(i, (t, 2)) for i in coords];
coords_ddot_syms = symbols('r1_dd r2_dd theta1_dd theta2_dd')
pos = [r1b, r1r, r2b, r2r]
equations = compute_equations_motion(g, k, m, l, coords, y, pos)

# Turn Eq(lhs, rhs) into lhs - rhs == 0
residuals = [eq.lhs - eq.rhs for eq in equations]

# Substitute second derivatives with dummy symbols (e.g. theta1_dd)
residuals_subs = [res.subs(dict(zip(d2, coords_ddot_syms))) for res in residuals]
from sympy.solvers.solveset import linear_eq_to_matrix

A, RHS = linear_eq_to_matrix(residuals_subs, coords_ddot_syms)
b = [b1b, b1r, b2b, b2r]
c = [c1b, c1r, c2b, c2r]
Q=gen_diss_force(b, c, l, pos, coords)

Delta = symbols("Delta")
subs_dict = {
    theta1 - theta2: -Delta,
    theta2 - theta1: Delta,
}
print("Simplified A with subs=")
A = simplify(A).subs(subs_dict)
novars = shape(A)[0];
for i in range(novars):
	print("A[" + str(i) + ",:] = " + str(A[i,:]))
	
print("Simplified b with subs=")
RHS = factor(simplify(RHS)).subs(subs_dict)
for i in range(novars):
	collRHS = collect(RHS[i], [M1, M2, mu1, mu2, k1, k2])
	print("b[" + str(i) + "] = " + str(collRHS))	
