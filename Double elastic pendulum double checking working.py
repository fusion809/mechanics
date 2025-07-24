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
v2bI_sq = dr2**2 + r2**2*dtheta2**2;
Delta = theta2 - theta1;
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

Delv21I_sq = simplify(v2b_sq - v1b_sq - v2bI_sq)
coords = [r1, r2, theta1, theta2];
dcoords = [dr1, dr2, dtheta1, dtheta2];
subs_dict = {
    theta1 - theta2: -Delta,
    theta2 - theta1: Delta,
}
for i in range(len(coords)):
	print(str(coords[i]) + " v1b_sq")
	print("partial/partial " + str(coords[i]) + " v1b_sq = ")
	dv1b_sq = diff(v1b_sq, coords[i])
	print(simplify(dv1b_sq).subs(subs_dict))
	print("partial/partial " + str(dcoords[i]) + " v1b_sq = ")
	ddv1b_sq = diff(v1b_sq, dcoords[i])
	print(simplify(ddv1b_sq).subs(subs_dict))
	print("time deriv of partial/partial " + str(dcoords[i]) + " v1b_sq = ")
	dddv1b_sq = diff(ddv1b_sq, t)
	print(simplify(dddv1b_sq).subs(subs_dict))
	print("delta'_" + str(coords[i]) + "v1b_sq=")
	print(simplify(dddv1b_sq - dv1b_sq).subs(subs_dict))
	print("\n")
	
print("Now for v2bI_sq")

for i in range(len(coords)):
	print(str(coords[i]) + " v2bI_sq")
	print("partial/partial " + str(coords[i]) + " v2bI_sq = ")
	dv2bI_sq = diff(v2bI_sq, coords[i])
	print(simplify(dv2bI_sq).subs(subs_dict))
	print("partial/partial " + str(dcoords[i]) + " v2bI_sq = ")
	ddv2bI_sq = diff(v2bI_sq, dcoords[i])
	print(simplify(ddv2bI_sq).subs(subs_dict))
	print("time deriv of partial/partial " + str(dcoords[i]) + " v2bI_sq = ")
	dddv2bI_sq = diff(ddv2bI_sq, t)
	print(simplify(dddv2bI_sq).subs(subs_dict))
	print("delta'_" + str(coords[i]) + "v2bI_sq=")
	print(simplify(dddv2bI_sq - dv2bI_sq).subs(subs_dict))
	print("\n")
	
print("Now for Delv21I_sq")

for i in range(len(coords)):
	print(str(coords[i]) + " Delv21I_sq")
	print("partial/partial " + str(coords[i]) + " Delv21I_sq = ")
	dDelv21I_sq = diff(Delv21I_sq, coords[i])
	print(simplify(dDelv21I_sq.replace(lambda x: x == theta1 - theta2, lambda x: -Delta)).subs(subs_dict))
	print("partial/partial " + str(dcoords[i]) + " Delv21I_sq = ")
	ddDelv21I_sq = diff(Delv21I_sq, dcoords[i])
	print(simplify(ddDelv21I_sq.replace(lambda x: x == theta1 - theta2, lambda x: -Delta)).subs(subs_dict))
	print("time deriv of partial/partial " + str(dcoords[i]) + " Delv21I_sq = ")
	dddDelv21I_sq = diff(ddDelv21I_sq, t)
	print(simplify(dddDelv21I_sq.replace(lambda x: x == theta1- theta2, lambda x: -Delta)).subs(subs_dict))
	print("delta'_" + str(coords[i]) + "Delv21I_sq=")
	print(simplify(dddDelv21I_sq.replace(lambda x: x == theta1 - theta2, lambda x: -Delta) - dDelv21I_sq.replace(lambda x: x == theta1 - theta2, lambda x: -Delta)).subs(subs_dict))
	print("\n")