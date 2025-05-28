import Pkg
Pkg.add("Gridap")
Pkg.add("LineSearches")
using Gridap, LineSearches

"""
Use a fixed point approach to solve a thermoforming quasi-variational inequality.
We use the Gridap FEM package in Julia and continuous piecewise linear
FEM for the u and T.
"""
domain = (0,1,0,1)
partition = (150,150)
model = CartesianDiscreteModel(domain,partition)
model = simplexify(model)
labels = get_face_labeling(model)
reffe_u = ReferenceFE(lagrangian,Float64, 1)

Vu = TestFESpace(model,reffe_u,labels=labels,dirichlet_tags="boundary",conformity=:H1)
VT = TestFESpace(model,reffe_u,labels=labels,conformity=:H1)
Uu = TrialFESpace(Vu, 0.0)
UT = TrialFESpace(VT)


Ω = Triangulation(model)
Γ = BoundaryTriangulation(model,labels)
dΩ = Measure(Ω,11)

Φ₀ = interpolate_everywhere(x->1.0 - 2*max(abs(x[1]-1/2),  abs(x[2]-1/2)), VT)
φ = interpolate_everywhere(x->sin(π*x[1])*sin(π*x[2]), VT)
fh = interpolate_everywhere(x->25, Vu)
# g and its derivative
q = 0.01
function g(s)
    if s ≤ 0
        return 1.0
    elseif 0 < s < q
        return (1-s/q)
    else
        return 0.0
    end
end
function dg(s)
    if s ≤ 0
        return 0.0
    elseif 0 < s < q
        return -1/q
    else
        return 0.0
    end
end

plus(x) = max(0, x)
plus²(x) = plus(x)^2
function dplus(x)
    if x>0
        return 1.0
    else
        return 0.0
    end
end

au(u, v) = ∫(∇(u) ⋅ ∇(v)  - fh ⋅ v + γ*(plus ∘ (u - (Φ₀ + φ ⋅ Th))) ⋅ v) * dΩ
jacu(u, du, v) =∫( ∇(du) ⋅ ∇(v) + γ*(dplus ∘ (u - (Φ₀ + φ ⋅ Th))) ⋅ du ⋅ v) * dΩ
aT(T, R) =∫( ∇(T) ⋅ ∇(R) + T ⋅ R  - (g ∘ (Φ₀ + φ ⋅ T - uh)) ⋅ R) * dΩ
jacT(T, dT, R) =∫( ∇(dT) ⋅ ∇(R) + dT ⋅ R - ((dg ∘ (Φ₀ + φ ⋅ T - uh)) ⋅ (φ ⋅ dT)) ⋅ R) * dΩ

au0(u, v) = ∫(∇(u) ⋅ ∇(v) + u ⋅ v) * dΩ
jacu0(u, du, v) =∫(∇(du) ⋅ ∇(v) + du ⋅ v) * dΩ

opu = FEOperator(au, jacu, Uu, Vu)
opT = FEOperator(aT, jacT, UT, VT)

energy(u) = ∫(0.5*∇(u) ⋅ ∇(u) - fh ⋅ u) * dΩ
penalty(u) = ∫(γ/2*(plus² ∘ (u - (Φ₀ + φ ⋅ Th)))) * dΩ
function update_γ(γ, k)
    infeasibility = sum(penalty(uh))
    functional = sum(energy(uh))
    print("         infeasibility: $infeasibility, functional: $functional.\n")
    Eₖ = γ * infeasibility / functional
    θₖ = functional + infeasibility
    C₂ₖ = Eₖ * (Eₖ + γ) * θₖ / γ
    C₁ₖ = C₂ₖ / Eₖ
    τₖ = 1/k
    return C₂ₖ /(τₖ  * abs(C₁ₖ - θₖ)) - Eₖ
end

function path_following_solve(opu, solver_u, u, J, γ)
    newton_its_u = 0
    u_ = u.free_values[:]
    for k in 1:100
        print("         Considering γ=$γ.\n")
        u, its = solve!(u,solver_u,opu)
        newton_its_u += its.result.iterations
        d = uh.free_values[:] - u_
        cauchy = sqrt(d' * J * d)
        print("         Cauchy norm: $(cauchy[end]).\n")
        u_ = u.free_values[:]

        if cauchy[end] < 1e-5
            break
        end

        global γ = update_γ(γ, k+1)

        if γ > 1e11
            break
        end
    end
    return u, newton_its_u
end

J = Gridap.Algebra.jacobian( FEOperator(au0, jacu0, Uu, Vu), interpolate_everywhere(x->0.0, Uu))

global uh = FEFunction(Vu, zeros(Vu.nfree))
global uh_ = uh.free_values[:]
global Th = FEFunction(VT, ones(VT.nfree))

nls_u = NLSolver(show_trace=false, method=:newton, linesearch=LineSearches.BackTracking(c_1=-1e8), ftol=1e-5, xtol=10*eps())
solver_u = FESolver(nls_u)

nls_T = NLSolver(show_trace=false, method=:newton, linesearch=LineSearches.BackTracking(c_1=-1e8), ftol=1e-10, xtol=10*eps())
solver_T = FESolver(nls_T)

global newton_its_T = 0 
global newton_its_u = 0
cauchy = []

tic = @elapsed for j in 1:10_000
    print("Considering Fixed Point Iteration: $j. \n")
    print("     Solving for T.\n")
    global Th, its = solve!(Th,solver_T,opT)
    global newton_its_T += its.result.iterations

    print("     Solving for u.\n")
    global γ = 1e0
    global uh, its_u = path_following_solve(opu, solver_u, uh, J, γ)
    global newton_its_u += its_u
    d = uh.free_values[:] - uh_
    push!(cauchy, sqrt(d' * J * d))
    print("Cauchy norm: $(cauchy[end]).\n")
    global uh_ = uh.free_values[:]
    if cauchy[end] < 1e-5
        break
    end
end
print("Run time(s): $tic, fixed point its: $(length(cauchy)), linear system solves: $(newton_its_T+newton_its_u)")