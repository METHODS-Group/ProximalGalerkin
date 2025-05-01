import Pkg
Pkg.add("Gridap")
Pkg.add("https://github.com/ioannisPApapadopoulos/SemismoothQVIs.jl.git")
using Gridap, SemismoothQVIs

"""
Use a semismooth active set strategy to solve a thermoforming quasi-variational 
inequality. We use the Gridap FEM package and SemismoothQVIs package  in Julia and 
continuous piecewise linear FEM for the u and T.

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
dΩ = Measure(Ω,11)

k = 1
Φ₀ = interpolate_everywhere(x->1.0 - 2*max(abs(x[1]-1/2),  abs(x[2]-1/2)), VT)
Ψ₀ = Φ₀
ϕ = interpolate_everywhere(x->sin(π*x[1])*sin(π*x[2]), UT)
ψ = ϕ
f = interpolate_everywhere(x->25, Uu)

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

Q = GeneralizedThermoformingQVI(dΩ, k, Φ₀, ϕ, Ψ₀, ψ, g, dg, f, Uu, UT)
u₀ = FEFunction(Vu, zeros(Vu.nfree))
T₀ = FEFunction(VT, ones(VT.nfree))

tic = @elapsed (zhs3, h1_3, its_3, is_3) = semismoothnewton(Q, u₀, T₀; ρ_min=1e-4, max_its=10, in_tol=1e-6, out_tol=1e-5, globalization=false, PF=true, bt=true, show_inner_trace=true);
print("Run time(s): $tic, active set its: $(its_3[1]), linear system solves: $(sum(its_3[2:4]))")