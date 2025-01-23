using LinearAlgebra, SparseArrays
import ClassicalOrthogonalPolynomials: band
using Plots, LaTeXStrings


f(x, y) = 0.0
function φ(x,y)
    r = sqrt(x^2 + y^2)
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
function residual(x::AbstractVector{T}, α::T, A::AbstractMatrix{T}, (fv, φv, w)::NTuple{3, <:AbstractVector{T}}, bcs::AbstractVector{Int}, n::Int) where T
    u = x[1:n]
    ψ = x[n+1:end]
    g = [α*A*u + ψ - α*fv - w; u - exp.(ψ) - φv]
    g[bcs].=0
    return g
end

function jacobian(α::T, A::AbstractMatrix{T}, Iden2::AbstractMatrix{T}, ψ::AbstractVector{T}, bcs::AbstractVector{Int}) where T
    J = [α*A Iden2;Iden2 -Diagonal(exp.(ψ))]
    J[:,bcs].=0
    J[bcs,:].=0
    view(J,band(0))[bcs] .= 1
    return J 
end

function fd_lvpp_solve(N::Int)
    xx = range(-1,1,N)
    A1 = sparse(Symmetric(Bidiagonal(2*ones(N), -ones(N-1), :U)) .* (N-1)^2)

    BC = zeros(size(A1))
    BC[1,1]=NaN
    BC[end,end]=NaN
    Iden = Diagonal(ones(size(A1,1)))
    BC = kron(BC,Iden) + kron(Iden,BC)
    bc = findall(isnan.(BC))
    bcs = unique([bc[i][2] for i in 1:lastindex(bc)])

    A = kron(A1,Iden) + kron(Iden,A1)
    n = size(A,1)
    Iden2 = Diagonal(ones(n))


    fv = Vector(vec(f.(xx,xx')))
    φv = Vector(vec(φ.(xx,xx')))

    ψ = zeros(n)
    w = zeros(n)
    u = zeros(n)
    u_ = zeros(n)

    α = 0.0
    C = 0.1
    r = 1.5
    q = 1.5

    newton_its = 0
    for k = 0:25
        α = min(max(C*r^(q^k) - α, C), 1e3)
        print("α = $α.\n")
        b = -residual([u;ψ], α, A, (fv, φv, w), bcs, n)
        print("Iteration 0, residual: $(norm(b)).\n")
        for iter = 1:2
            J = jacobian(α, A, Iden2, ψ, bcs)

            dz = J \ b

            u = u + dz[1:n]
            ψ = ψ + dz[n+1:end]

            newton_its += 1
            b = -residual([u;ψ], α, A, (fv, φv, w), bcs, n)
            normres = norm(b)
            print("Iteration $iter, residual: $(normres).\n")
            if normres < 1e-4
                break
            end
        end
        w = copy(ψ)

        if norm(u-u_) < 1e-9
            break
        else
            u_ = copy(u)
        end
    end
    return xx, reshape(u, N, N), reshape(φv, N, N), newton_its
end

its = Int[]

for j in 2:7
    N = 2^j + 1
    xx, U, Φ, newton_its =  fd_lvpp_solve(N)
    push!(its, newton_its)
end

Plots.surface(xx,xx,Φ,color=:greys, zlim=[0,0.6], cbar=:none)
Plots.surface!(xx,xx,U, zlim=[0,0.6], color=:diverging, fillalpha=0.9)