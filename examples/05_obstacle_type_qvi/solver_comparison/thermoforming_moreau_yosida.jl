
import Pkg
Pkg.add("Gridap")
Pkg.add("LineSearches")
using Gridap, LineSearches

"""
Use a Moreau-Yosida penalization to solve a thermoforming quasi-variational inequality.
We use the Gridap FEM package in Julia and continuous piecewise linear
FEM for the u and T.

"""
function copy_z(zh::Gridap.MultiField.MultiFieldFEFunction)
    FEFunction(zh.fe_space, zh.free_values[:])
end

domain = (0,1,0,1)
partition = (150,150)
model = CartesianDiscreteModel(domain,partition)
model = simplexify(model) # Use simplex elements

labels = get_face_labeling(model)
# CG1 FEM
reffe_u = ReferenceFE(lagrangian,Float64, 1)
reffe_ψ = ReferenceFE(lagrangian,Float64, 1)

Vu = TestFESpace(model,reffe_u,labels=labels,dirichlet_tags="boundary",conformity=:H1)
VT = TestFESpace(model,reffe_u,labels=labels,conformity=:H1)
Uu = TrialFESpace(Vu, 0.0) # zero bcs
UT = TrialFESpace(VT)

V = MultiFieldFESpace([Vu, VT]) # u, T
U = MultiFieldFESpace([Uu, UT]) # u, T

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model,labels)
dΩ = Measure(Ω,11)

# Initial mould
Φ₀ = interpolate_everywhere(x->1.0 - 2*max(abs(x[1]-1/2),  abs(x[2]-1/2)), VT)
# Smoothing function
φ = interpolate_everywhere(x->sin(π*x[1])*sin(π*x[2]), UT)
# Forcing term
fh = interpolate_everywhere(x->25, Uu) # forcing term

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

# Residual forms
a((u, T), (v, R)) =
     ∫( 
        ∇(u) ⋅ ∇(v) - fh ⋅ v + γ*(plus ∘ (u - (Φ₀ + φ ⋅ T))) ⋅ v
        + ∇(T) ⋅ ∇(R) + T ⋅ R  - (g ∘ (Φ₀ + φ ⋅ T - u)) ⋅ R
     ) * dΩ

# Jacobian
jac((u, T), (du, dT), (v, R)) =
     ∫( 
         ∇(du) ⋅ ∇(v) + γ*(dplus ∘ (u - (Φ₀ + φ ⋅ T))) ⋅ du ⋅ v - γ*(dplus ∘ (u - (Φ₀ + φ ⋅ T))) ⋅ φ ⋅ dT ⋅ v
         + ∇(dT) ⋅ ∇(R) + dT ⋅ R - (dg ∘ (Φ₀ + φ ⋅ T - u)) ⋅ φ ⋅ dT ⋅ R + (dg ∘ (Φ₀ + φ ⋅ T - u)) ⋅ du ⋅ R
     ) * dΩ


# Get matrix for H1-inner products
au0(uh, v) =∫(∇(uh) ⋅ ∇(v) + uh ⋅ v) * dΩ
jacu0(uh, duh, v) =∫(∇(duh) ⋅ ∇(v) + duh ⋅ v) * dΩ
J = Gridap.Algebra.jacobian( FEOperator(au0, jacu0, Uu, Vu), interpolate_everywhere(x->0.0, Uu))

# γ-update scheme
energy(u) = ∫(0.5*∇(u) ⋅ ∇(u) - fh ⋅ u) * dΩ
penalty(u, T) = ∫(γ/2*(plus² ∘ (u - (Φ₀ + φ ⋅ T)))) * dΩ
function update_γ(zh, γ, k)
    uh, Th = zh.single_fe_functions
    infeasibility = sum(penalty(uh, Th))
    functional = sum(energy(uh))
    print("         infeasibility: $infeasibility, functional: $functional.\n")
    Eₖ = γ * infeasibility / functional
    θₖ = functional + infeasibility
    C₂ₖ = Eₖ * (Eₖ + γ) * θₖ / γ
    C₁ₖ = C₂ₖ / Eₖ
    τₖ = 1/k
    return C₂ₖ /(τₖ  * abs(C₁ₖ - θₖ)) - Eₖ
end

# Initial guess
u0 = interpolate_everywhere(x->0.0, Uu).free_values
T0 = interpolate_everywhere(x->1.0, UT).free_values
global zh = FEFunction(U,[u0; T0]);
newton_its=[];

cauchy =  []
global zh_ = copy_z(zh)
op = FEOperator(a, jac, U, V)
nls = NLSolver(show_trace=true, method=:newton, linesearch=LineSearches.BackTracking(c_1=-1e8), ftol=1e-5, xtol=10*eps())
solver = FESolver(nls)

# Run Moreau-Yosida path-following solve
global γ = 1.0
tic = @elapsed for j in 1:100
    print("Considering γ = $γ. \n")

    # Newton solver with backtracking linesearch
    global zh, its = solve!(zh,solver,op)
    its = its.result.iterations

    push!(newton_its, its)

    d = zh.single_fe_functions[1].free_values[:] - zh_.single_fe_functions[1].free_values[:]
    push!(cauchy, sqrt(d' * J * d))
    print("Cauchy norm: $(cauchy[end]).\n")
    global zh_ = copy_z(zh)

    # γ-update rule
    global γ = update_γ(zh, γ, j+1)

    # break if tolerance reached
    if cauchy[end] < 1e-5
        break
    end
end
print("Run time(s): $tic, moreau-yosida its: $(length(newton_its)), linear system solves: $(sum(newton_its))")