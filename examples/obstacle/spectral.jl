using ClassicalOrthogonalPolynomials, MultivariateOrthogonalPolynomials
import MultivariateOrthogonalPolynomials: ZernikeITransform
using SparseArrays, LinearAlgebra
using IterativeSolvers, LinearMaps, MatrixFactorizations
using Plots, LaTeXStrings

"""
Solve the obstacle problem on a disk domain with LVPP 
discretized with a sparse spectral method via Zernike polynomials.
"""

T = Float64

get_rs(x) = x.r
get_θs(x) = x.θ

# Forcing term
f(r, θ) = 0.0

function φ(r, θ)
    # Obstacle in polar coordinates
    r0 = 0.5
    β = 0.9
    b = r0*β
    t = sqrt(r0^2 - b^2)
    B = t + b^2/t
    C = -b/t
    if r > b
        return B + C * r
    else
        return sqrt(r0^2 - r^2)
    end
end


Z = Zernike(1) # Zernike(1) quasi-matrix
wZ = Weighted(Z) # (1-r^2)Zernike(1) quasi-matrix
ΔF = (Z \ (Laplacian(axes(Z,1)) * wZ)); # Laplacian matrix
BF = (Z \ wZ) ./ 2; # bug in lowering conversion needs to be corrected with /2.

function residual(x::AbstractVector{T}, α::T, (plZ,iplZ), (Δ,B)::NTuple{2, <:AbstractMatrix{T}}, (fv,φv,w)::NTuple{3, <:AbstractVector}, n::Int) where T
    u = x[1:n]
    ψ = x[n+1:end]
    # The nonlinear component is computed by transforming to physical space
    # applying the exponential and transforming back to coefficient space
    [-α*Δ*u + ψ - α*fv - w; B*u - plZ*(exp.(iplZ*ψ)) - φv]
end

function apply_jacobian(x::AbstractVector{T}, α::T, (plZ,iplZ), (Δ,B)::NTuple{2, <:AbstractMatrix{T}}, ψ::AbstractVector{T}, n::Int) where T
    du = x[1:n]
    dψ = x[n+1:end]
    # The nonlinear component is computed by transforming to physical space
    # applying the exponential and transforming back to coefficient space
    [-α*Δ*du + dψ; B*du - plZ*(exp.(iplZ*ψ) .* (iplZ*dψ))]
end

# LVPP solver
function zernike_lvpp_solve(ΔF, BF, p::Int)

    # Extract out finite matrices
    KR = Block.(1:p+1)
    Δ = sparse(ΔF[KR,KR])
    B = sparse(BF[KR,KR])
    n = size(Δ, 1)
    Iden = Diagonal(ones(n))

    # Transform coefficient space -> physical space
    plZ =  plan_transform(Z, Block(p))
    # Transform physical space -> coefficient space
    iplZ = ZernikeITransform{T}(p, 0, 1)
    g = ClassicalOrthogonalPolynomials.grid(Z, Block(p))

    # Find coefficient expansions of forcing term and obstacle
    fv = Vector(plZ *  f.(get_rs.(g), get_θs.(g)))
    φv = Vector(plZ * φ.(get_rs.(g), get_θs.(g)))

    ψ, w, u, u_ = zeros(n), zeros(n), zeros(n), ones(n)


    newton_its, gmres_its = 0, 0

    # Parameters for α-update
    α, C, r, q = 0.1, 0.1, 1.5, 1.5

    for k = 1:100
        print("α = $α.\n")
        α = min(max(C*r^(q^k) - α, C), 1e3)
        b = -residual([u;ψ], α, (plZ,iplZ), (Δ,B), (fv, φv, w), n)
        print("Iteration 0, residual: $(norm(b)).\n")

        # Cap each LVPP subproblem solve at 2 Newton iterations
        for iter = 1:2
            apply_jac(x) = apply_jacobian(x, α, (plZ,iplZ), (Δ,B), ψ, n)
            J = LinearMap(apply_jac, 2*n; ismutating=false)

            c = (plZ*(exp.(iplZ * ψ)))[1]
            P = [-α*Δ Iden; B -Iden]
            lu_P = MatrixFactorizations.lu(P)

            # Solve linear system with a preconditioned GMRES solver
            dz, info = IterativeSolvers.gmres(J, b, Pl=lu_P, restart=n, log=true)
            gmres_its += info.iters

            u = u + dz[1:n]
            ψ = ψ + dz[n+1:end]

            newton_its += 1
            b = -residual([u;ψ], α, (plZ,iplZ), (Δ,B), (fv, φv, w), n)
            normres = norm(b)
            print("Iteration $iter, GMRES iters: $(info.iters), residual: $(normres).\n")
            if normres < 1e-4
                break
            end
        end
        w = copy(ψ)

        # Break if we reach the tolerance
        if norm(u-u_) < 1e-9
            break
        else
            u_ = copy(u)
        end
    end
    return u, ψ, φv, (newton_its, gmres_its)
end


its = Int[]

# Increasing degree discretizaions
for p in 8:8:48
    @assert(iseven(p))
    u, ψ, φv, (newton_its, gmres_its) = zernike_lvpp_solve(ΔF, BF, p);
    push!(its, newton_its)
end

# For plotting
function evaluate(Q,u,x,y)
    if x^2 + y^2 ≤ 1
        (Q[:,Block.(1:p+1)]*u)[[x,y]]
    elseif 1 < x^2+y^2 < 1.01
        return 0.0
    else
        return -0.01
    end
end

if false
    evaluate_u(x,y) = evaluate(wZ,u,x,y)
    evaluate_φ(x,y) = evaluate(Z,φv,x,y)
    xx = range(-1,1,101)

    Ux= evaluate_u.(xx,xx')
    Φx= evaluate_φ.(xx,xx')

    Plots.surface(xx,xx,Φx,color=:greys, zlim=[0,0.7], cbar=:none)
    Plots.surface!(xx,xx,Ux, zlim=[0,0.7], color=:diverging, fillalpha=0.9)
    Plots.savefig("spectral-obstacle.png")

    Ux= evaluate_u.(xx,[0.0])
    Φx= evaluate_φ.(xx,[0.0])
    Plots.plot(xx, Ux)
    Plots.plot!(xx, Φx)
end