
import Pkg
Pkg.add("Gridap")
Pkg.add("LineSearches")
Pkg.add("Plots")
Pkg.add("LaTeXStrings")
using Gridap, LineSearches
using Plots, LaTeXStrings

"""
Use LVPP to solve a thermoforming quasi-variational inequality.
We use the Gridap FEM package in Julia and continuous piecewise linear
FEM for the u, T, and ψ.

"""
domain = (0,1,0,1)
partition = (150,150)
model = CartesianDiscreteModel(domain,partition)
# Use simplex elements
model = simplexify(model)

labels = get_face_labeling(model)
# CG1 FEM
reffe_u = ReferenceFE(lagrangian,Float64, 1)
reffe_ψ = ReferenceFE(lagrangian,Float64, 1)

Vu = TestFESpace(model,reffe_u,labels=labels,dirichlet_tags="boundary",conformity=:H1)
VT = TestFESpace(model,reffe_u,labels=labels,conformity=:H1)
Uu = TrialFESpace(Vu, 0.0) # zero bcs
UT = TrialFESpace(VT)

Vψ = TestFESpace(model,reffe_ψ,labels=labels,conformity=:H1)
Uψ = TrialFESpace(Vψ)

V = MultiFieldFESpace([Vu, VT, Vψ]) # u, T, ψ
U = MultiFieldFESpace([Uu, UT, Uψ]) # u, T, ψ

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

# Residual forms
a((uh, Th, ψh), (v, R, ϕ), α, wh) =
     ∫( 
        α*∇(uh) ⋅ ∇(v) +  ψh ⋅ v - α*fh ⋅ v - wh ⋅ v
        + ∇(Th) ⋅ ∇(R) + Th ⋅ R  - (g ∘ (exp ∘ (-ψh))) ⋅ R
        + uh ⋅ ϕ + (exp ∘ (-ψh)) ⋅ ϕ - (Φ₀ + φ ⋅ Th) ⋅ ϕ
     ) * dΩ

# Jacobian
jac((uh, Th, ψh), (duh, dTh, dψh), (v, R, ϕ), α) =
     ∫( 
         α*∇(duh) ⋅ ∇(v) + dψh ⋅ v
         + ∇(dTh) ⋅ ∇(R) + dTh ⋅ R - (dg ∘ (exp ∘ (-ψh))) ⋅ (exp ∘ (-ψh)) ⋅ (-dψh) ⋅ R
         + duh ⋅ ϕ - ((exp ∘ (-ψh)) ⋅ dψh ) ⋅ ϕ - (φ ⋅ dTh) ⋅ ϕ - 1e-10/α*(∇(dψh) ⋅ ∇(ϕ))
     ) * dΩ


# Get matrix for H1-inner products
au0(uh, v) =∫(∇(uh) ⋅ ∇(v) + uh ⋅ v) * dΩ
jacu0(uh, duh, v) =∫(∇(duh) ⋅ ∇(v) + duh ⋅ v) * dΩ
J = Gridap.Algebra.jacobian( FEOperator(au0, jacu0, Uu, Vu), interpolate_everywhere(x->0.0, Uu))

# Initial guess
u0 = interpolate_everywhere(x->0.0, Uu).free_values
T0 = interpolate_everywhere(x->1.0, UT).free_values
w0 = interpolate_everywhere(x->0.0, Uψ).free_values
zh = FEFunction(U,[u0; T0; w0]);
wh = FEFunction(Uψ,w0);
newton_its=[];

zhs, cauchy = [zh.single_fe_functions[1].free_values[:]], []

# Run LVPP solve
α = 2^(-6)
for j in 1:100
    print("Considering α = $α. \n")

    b((uh, Th, ψh), (v, R, ϕ)) = a((uh, Th, ψh), (v, R, ϕ), α, wh)
    jb((uh, Th, ψh), (duh, dTh, dψh), (v, R, ϕ)) = jac((uh, Th, ψh), (duh, dTh, dψh), (v, R, ϕ), α)
    op = FEOperator(b, jb, U, V)

    # Newton solver with backtracking linesearch
    nls = NLSolver(show_trace=true, method=:newton, linesearch=LineSearches.BackTracking(c_1=-1e8), ftol=1e-5, xtol=10*eps())
    solver = FESolver(nls)
    zh, its = solve!(zh,solver,op)
    its = its.result.iterations

    push!(zhs, zh.single_fe_functions[1].free_values[:])
    wh = FEFunction(Vψ, zh.single_fe_functions[3].free_values)
    push!(newton_its, its)

    d = zhs[end] - zhs[end-1]
    push!(cauchy, sqrt(d' * J * d))
    print("Cauchy norm: $(cauchy[end]).\n")

    # α-update rule
    α = 4*α

    # break if tolerance reached
    if cauchy[end] < 1e-9
        break
    end
end

# Save solutions for Paraview
uh = zh.single_fe_functions[1]
writevtk(Ω,"membrane", cellfields=["membrane"=>uh])
Th = zh.single_fe_functions[2]
mould = interpolate_everywhere(Φ₀ + φ ⋅ Th, VT)
writevtk(Ω,"mould", cellfields=["mould"=>mould])


# Plot 1D slice of solution
xx = range(0,1,101)
p = plot(xx, [uh(Point.(xx,0.5)) mould(Point.(xx,0.5)) Φ₀(Point.(xx, 0.5)) Th(Point.(xx,0.5))],
    linewidth=2,
    label=["Membrane" "Mould" "Original Mould" "Temperature"],
    linestyle=[:solid :dash],
    xlabel=L"x",
    xlabelfontsize=20, xtickfontsize=12,ytickfontsize=12,
)
Plots.savefig("thermoforming-slice.pdf")