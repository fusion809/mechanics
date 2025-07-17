using Symbolics, LinearAlgebra
@variables t
@variables m1b m2b m1r m2r l1 l2 k1 k2 g b1r b2r b1b b2b c1r c1b c2r c2b
@syms r1(t) θ1(t) r2(t) θ2(t)
@variables Qr1, Qr2, Qθ1, Qθ2

# Vector components as tuples (Symbolics doesn't have built-in vector algebra like sympy.vector)
function vec(x, y)
    return [x, y]
end

# First derivatives
D = Differential(t)
dr1 = D(r1)
dθ1 = D(θ1)
dr2 = D(r2)
dθ2 = D(θ2)

# ---------------------------
# Position and velocity vectors
# ---------------------------
x1b = r1(t)*cos(θ1(t)); y1b = r1(t)*sin(θ1(t))
vx1b = expand_derivatives(D(x1b)); vy1b = expand_derivatives(D(y1b))
v1b = vec(vx1b, vy1b)
v1b_sq = simplify(vx1b^2 + vy1b^2)
v1b_mag = sqrt(v1b_sq)

x2b = x1b + r2(t)*cos(θ2(t)); y2b = y1b + r2(t)*sin(θ2(t))
vx2b = expand_derivatives(D(x2b)); vy2b = expand_derivatives(D(y2b))
v2b = vec(vx2b, vy2b)
v2b_sq = simplify(vx2b^2 + vy2b^2)
v2b_mag = sqrt(v2b_sq)

x1r = x1b / 2; y1r = y1b / 2
vx1r = expand_derivatives(D(x1r)); vy1r = expand_derivatives(D(y1r))
v1r = vec(vx1r, vy1r)
v1r_sq = simplify(vx1r^2 + vy1r^2)
v1r_mag = sqrt.(v1r_sq)

x2r = x1b + r2(t)*cos(θ2(t))/2; y2r = y1b + r2(t)*sin(θ2(t))/2
vx2r = expand_derivatives(D(x2r)); vy2r = expand_derivatives(D(y2r))
v2r = vec(vx2r, vy2r)
v2r_sq = simplify(vx2r^2 + vy2r^2)
v2r_mag = sqrt.(v2r_sq)

# ---------------------------
# Generalized force vectors
# ---------------------------
function scaled_force(v, b, c, vmag)
    return simplify.(-(b + c*abs(vmag)) .* v)
end

Frod1 = scaled_force(v1r, b1r, c1r, v1r_mag)
Frod2 = scaled_force(v2r, b2r, c2r, v2r_mag)
Fbob1 = scaled_force(v1b, b1b, c1b, v1b_mag)
Fbob2 = scaled_force(v2b, b2b, c2b, v2b_mag)

# Partial derivatives (Jacobian-style)
coords = [r1, r2, θ1, θ2]
vars_flat = [r1(t), r2(t), θ1(t), θ2(t)]
function grad_vec(pos, wrt)
    return [expand_derivatives(Differential(wrt_i)(comp)) for comp in pos, wrt_i in wrt]
end

# Positions
r1b = vec(x1b, y1b)
r2b = vec(x2b, y2b)
r1r = vec(x1r, y1r)
r2r = vec(x2r, y2r)

# Gradients for dot products
e1b = grad_vec(r1b, vars_flat)
e2b = grad_vec(r2b, vars_flat)
e1r = grad_vec(r1r, vars_flat)
e2r = grad_vec(r2r, vars_flat)

# Compute virtual work contributions
Qs = Vector{Num}(undef, 4)
for i in 1:4
    Qs[i] = simplify(
        dot(Frod1, e1r[:, i]) + dot(Frod2, e2r[:, i]) +
        dot(Fbob1, e1b[:, i]) + dot(Fbob2, e2b[:, i])
    )
end
Qs = [Qr1, Qr2, Qθ1, Qθ2]

# ---------------------------
# Lagrangian
# ---------------------------
T = m1b/2 * v1b_sq + m2b/2 * v2b_sq + m1r/2 * v1r_sq + m2r/2 * v2r_sq +
    m1r/24 * r1(t)^2 * dθ1(t)^2 + m2r/24 * r2(t)^2 * dθ2(t)^2

V = m1b * g * y1b + m1r * g * r1(t) * y1r + m2b * g * y2b + m2r * g * y2r +
    k1/2 * (r1(t) - l1)^2 + k2/2 * (r2(t) - l2)^2

L = simplify(T - V)

# ---------------------------
# Equations of Motion (Parallelised)
# ---------------------------
D2t = Differential(t) * Differential(t)
accs = [D2t(r1(t)), D2t(r2(t)), D2t(θ1(t)), D2t(θ2(t))]

function lagrange_eqn(L, Q, q)
    dq = Differential(t)(q)
    dL_dq = expand_derivatives(Differential(q)(L))
    dL_ddq = expand_derivatives(Differential(dq)(L))
    dt_dL_ddq = expand_derivatives(Differential(t)(dL_ddq))
    lhs = dt_dL_ddq - dL_dq
    return simplify(lhs - Q)
end

# Parallel computation using Threads.@threads
eoms = Vector{Num}(undef, 4)
Threads.@threads for i in 1:4
    eoms[i] = lagrange_eqn(L, Qs[i], coords[i])
end

# Solve for second derivatives
sols = Symbolics.symbolic_linear_solve(eoms, accs)

# Display results
secdernames = ["d2r1", "d2r2", "d2theta1", "d2theta2"]
for i in 1:4
    println(secdernames[i], " = ")
    println(sols[accs[i]])
end