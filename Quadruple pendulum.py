from sympy import symbols, Function, diff, cos, sin, simplify, sqrt, Abs, Eq, solve, latex, integrate, factor
from sympy.vector import CoordSys3D
from multiprocessing import Pool, cpu_count
from math import floor
N = CoordSys3D('N');
# Set up symbols
t = symbols('t')
m1b = symbols('m1b'); 
m2b = symbols('m2b'); 
m3b = symbols('m2b'); 
m4b = symbols('m2b'); 
m1r = symbols('m1r'); 
m2r = symbols('m2r'); 
m3r = symbols('m2r'); 
m4r = symbols('m2r'); 
m = [m1b, m1r, m2b, m2r, m3b, m3r, m4b, m4r]
r1 = symbols('r1');
r2 = symbols('r2');
r3 = symbols("r3");
r4 = symbols("r4");
b1r = symbols('b1r'); 
b2r = symbols('b2r'); 
b3r = symbols('b3r'); 
b4r = symbols('b4r'); 
b1b = symbols('b1b'); 
b2b = symbols('b2b'); 
b3b = symbols('b3b'); 
b4b = symbols('b4b'); 
c1r = symbols("c1r"); 
c1b = symbols("c1b"); 
c2r = symbols("c2r"); 
c2b = symbols("c2b");
c3r = symbols("c3r"); 
c3b = symbols("c3b");
c4r = symbols("c4r"); 
c4b = symbols("c4b");
r1 = Function('r1')(t); 
dr1 = diff(r1, t);
theta1 = Function('theta1')(t); 
dtheta1 = diff(theta1, t);
theta2 = Function('theta2')(t); 
dtheta2 = diff(theta2, t);
theta3 = Function('theta3')(t);
dtheta3 = diff(theta3, t);
theta4 = Function('theta4')(t);
dtheta4 = diff(theta4, t);
coords = [theta1, theta2, theta3, theta4]
dcoords = [dtheta1, dtheta2, dtheta3, dtheta4]
Delta21, Delta31, Delta32, Delta41, Delta42, Delta43 = symbols("Delta21, Delta31, Delta32, Delta41, Delta42, Delta43");
subs_dict = {
	theta1-theta2: -Delta21,
	theta2-theta1: Delta21,
	theta1-theta3: -Delta31,
	theta3-theta1: Delta31,
	theta2-theta3: -Delta32,
	theta3-theta2: Delta32,
	theta1-theta4: -Delta41,
	theta4-theta1: Delta41,
	theta2-theta4: -Delta42,
	theta4-theta2: Delta42,
	theta3-theta4: -Delta43,
	theta4-theta3: Delta43,
}

x1b = r1*cos(theta1);
y1b = r1*sin(theta1);
dx1b = diff(x1b, t);
dy1b = diff(y1b, t);
x1r = x1b/2;
y1r = y1b/2;
dx1r = diff(x1r, t);
dy1r = diff(y1r, t);
r1b = x1b*N.i + y1b*N.j;
r1r = x1r*N.i + y1r*N.j;
v1b = dx1b*N.i + dy1b*N.j;
v1r = dx1r*N.i + dy1r*N.j;
v1b_sq = simplify(v1b.dot(v1b).subs(subs_dict), simultaneous=True);
v1b_sq = v1b_sq.subs(subs_dict)
v1r_sq = simplify(v1r.dot(v1r).subs(subs_dict), simultaneous=True);
v1r_sq = v1r_sq.subs(subs_dict)

x2b = r2*cos(theta2);
y2b = r2*sin(theta2);
x2r = x2b/2;
y2r = y2b/2;
x2b += x1b;
y2b += y1b;
x2r += x1b;
y2r += y1b;
dx2b = diff(x2b, t);
dy2b = diff(y2b, t);
dx2r = diff(x2r, t);
dy2r = diff(y2r, t);
r2b = x2b*N.i + y2b*N.j;
r2r = x2r*N.i + y2r*N.j;
v2b = dx2b*N.i + dy2b*N.j;
v2r = dx2r*N.i + dy2r*N.j;
v2b_sq = simplify(v2b.dot(v2b).subs(subs_dict), simultaneous=True);
v2b_sq = v2b_sq.subs(subs_dict)
v2r_sq = simplify(v2r.dot(v2r).subs(subs_dict), simultaneous=True);
v2r_sq = v2r_sq.subs(subs_dict)

x3b = r3*cos(theta3);
y3b = r3*sin(theta3);
x3r = x3b/2;
y3r = y3b/2;
x3b += x2b;
y3b += y2b;
x3r += x2b;
y3r += y2b;
dx3b = diff(x3b, t);
dy3b = diff(y3b, t);
dx3r = diff(x3r, t);
dy3r = diff(y3r, t);
r3b = x3b*N.i + y3b*N.j;
r3r = x3r*N.i + y3r*N.j;
v3b = dx3b*N.i + dy3b*N.j;
v3r = dx3r*N.i + dy3r*N.j;
v3b_sq = simplify(v3b.dot(v3b).subs(subs_dict), simultaneous=True);
v3b_sq = v3b_sq.subs(subs_dict)
v3r_sq = simplify(v3r.dot(v3r).subs(subs_dict), simultaneous=True);
v3r_sq = v3r_sq.subs(subs_dict)

x4b = r4*cos(theta4);
y4b = r4*sin(theta4);
x4r = x4b/2;
y4r = y4b/2;
x4b += x3b;
y4b += y3b;
x4r += x3b;
y4r += y3b;
dx4b = diff(x4b, t);
dy4b = diff(y4b, t);
dx4r = diff(x4r, t);
dy4r = diff(y4r, t);
r4b = x4b*N.i + y4b*N.j;
r4r = x4r*N.i + y4r*N.j;
v4b = dx4b*N.i + dy4b*N.j;
v4r = dx4r*N.i + dy4r*N.j;
v4b_sq = simplify(v4b.dot(v4b).subs(subs_dict), simultaneous=True);
v4b_sq = v4b_sq.subs(subs_dict)
v4r_sq = simplify(v4r.dot(v4r).subs(subs_dict), simultaneous=True);
v4r_sq = v4r_sq.subs(subs_dict)

vsq = [r1**2*dtheta1**2, r1**2*dtheta1**2/4, r1**2*dtheta1**2 + r2**2*dtheta2**2 + 2*r1*r2*dtheta1*dtheta2*cos(Delta21), r1**2*dtheta1**2 + r2**2*dtheta2**2/4 + r1*r2*dtheta1*dtheta2*cos(Delta21), r1**2*dtheta1**2 + r2**2*dtheta2**2 + r3**2*dtheta3**2 + 2*r1*r2*dtheta1*dtheta2*cos(Delta21) + 2*r1*r3*dtheta1*dtheta3*cos(Delta31) + 2*r3*r2*dtheta3*dtheta2*cos(Delta32), r1**2*dtheta1**2 + r2**2*dtheta2**2 + r3**2*dtheta3**2/4 + 2*r1*r2*dtheta1*dtheta2*cos(Delta21) + r1*r3*dtheta1*dtheta3*cos(Delta31) + r3*r2*dtheta3*dtheta2*cos(Delta32), r1**2*dtheta1**2 + r2**2*dtheta2**2 + r3**2*dtheta3**2 + r4**2*dtheta4**2 + 2*r1*r2*dtheta1*dtheta2*cos(Delta21) + 2*r1*r3*dtheta1*dtheta3*cos(Delta31) + 2*r3*r2*dtheta3*dtheta2*cos(Delta32) + 2*r1*r4*dtheta1*dtheta4*cos(Delta41) + 2*r4*r2*dtheta4*dtheta2*cos(Delta42) + 2*r4*r3*dtheta4*dtheta3*cos(Delta43), r1**2*dtheta1**2 + r2**2*dtheta2**2 + r3**2*dtheta3**2 + r4**2*dtheta4**2/4 + 2*r1*r2*dtheta1*dtheta2*cos(Delta21) + 2*r1*r3*dtheta1*dtheta3*cos(Delta31) + 2*r3*r2*dtheta3*dtheta2*cos(Delta32) + r1*r4*dtheta1*dtheta4*cos(Delta41) + r4*r2*dtheta4*dtheta2*cos(Delta42) + r4*r3*dtheta4*dtheta3*cos(Delta43)]
L = 0;
r = [r1, r2, r3, r4];
theta = [theta1, theta2, theta3, theta4];
dtheta = [dtheta1, dtheta2, dtheta3, dtheta4];
y = [y1b, y1r, y2b, y2r, y3b, y3r, y4b, y4r]
g = symbols("g");
for i in range(len(m)):
	L += m[i]/2 * vsq[i];
	idx = floor(i/2);
	if ((i-1) % 2 == 0):
		L += m[i] * r[round(idx)]**2 * dtheta[round(idx)]**2/24;
	L -= m[i] * g * y[i];
	
L = simplify(L)