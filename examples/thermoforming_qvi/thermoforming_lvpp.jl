using Gridap
using LineSearches
using Plots, LaTeXStrings

domain = (0,1,0,1)
partition = (150,150)
model = CartesianDiscreteModel(domain,partition)
model = simplexify(model)

labels = get_face_labeling(model)
# model = Triangulation(model)
p = 1
reffe_u = ReferenceFE(lagrangian,Float64, p)
reffe_ψ = ReferenceFE(lagrangian,Float64, p)

Vu = TestFESpace(model,reffe_u,labels=labels,dirichlet_tags="boundary",conformity=:H1)
VT = TestFESpace(model,reffe_u,labels=labels,conformity=:H1)
Uu = TrialFESpace(Vu, 0.0)
UT = TrialFESpace(VT)

Vψ = TestFESpace(model,reffe_ψ,labels=labels,conformity=:H1)
Uψ = TrialFESpace(Vψ)

V = MultiFieldFESpace([Vu, VT, Vψ]) # u, T, ψ
U = MultiFieldFESpace([Uu, UT, Uψ]) # u, T, ψ

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model,labels)

dΩ = Measure(Ω,p+10)
Φ₀ = interpolate_everywhere(x->1.0 - 2*max(abs(x[1]-1/2),  abs(x[2]-1/2)), VT)
# Φ₀ = interpolate_everywhere(x->1.0 - 2*abs(x[1]-1/2), VT)
φ = interpolate_everywhere(x->sin(π*x[1])*sin(π*x[2]), UT)
# φ = interpolate_everywhere(x->sin(π*x[1]), UT)
fh = interpolate_everywhere(x->25, Uu)
# writevtk(Ω,"smooth", cellfields=["smooth"=>φ])
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

a((uh, Th, ψh), (v, R, ϕ), α, wh) =
     ∫( 
        α*∇(uh) ⋅ ∇(v) +  ψh ⋅ v - α*fh ⋅ v - wh ⋅ v
        + ∇(Th) ⋅ ∇(R) + Th ⋅ R  - (g ∘ (exp ∘ (-ψh))) ⋅ R
        + uh ⋅ ϕ + (exp ∘ (-ψh)) ⋅ ϕ - (Φ₀ + φ ⋅ Th) ⋅ ϕ
     ) * dΩ

jac((uh, Th, ψh), (duh, dTh, dψh), (v, R, ϕ), α) =
     ∫( 
         α*∇(duh) ⋅ ∇(v) + dψh ⋅ v
         + ∇(dTh) ⋅ ∇(R) + dTh ⋅ R - (dg ∘ (exp ∘ (-ψh))) ⋅ (exp ∘ (-ψh)) ⋅ (-dψh) ⋅ R
         + duh ⋅ ϕ - ((exp ∘ (-ψh)) ⋅ dψh ) ⋅ ϕ - (φ ⋅ dTh) ⋅ ϕ - 1e-10/α*(∇(dψh) ⋅ ∇(ϕ))
     ) * dΩ

# a((uh, Th, ψh), (v, R, ϕ), α, wh) =
#      ∫( 
#         α*∇(uh) ⋅ ∇(v) +  ψh ⋅ v - α*fh ⋅ v - wh ⋅ v
#         + ∇(Th) ⋅ ∇(R) + Th ⋅ R  - (g ∘ (Φ₀ + φ ⋅ Th - uh)) ⋅ R
#         + uh ⋅ ϕ + (exp ∘ (-ψh)) ⋅ ϕ - (Φ₀ + φ ⋅ Th) ⋅ ϕ
#      ) * dΩ

# jac((uh, Th, ψh), (duh, dTh, dψh), (v, R, ϕ), α) =
#      ∫( 
#          α*∇(duh) ⋅ ∇(v) + dψh ⋅ v
#          + ∇(dTh) ⋅ ∇(R) + dTh ⋅ R - (dg ∘ (Φ₀ + φ ⋅ Th - uh)) ⋅ (φ ⋅ dTh - duh) ⋅ R
#          + duh ⋅ ϕ - ((exp ∘ (-ψh)) ⋅ dψh ) ⋅ ϕ - (φ ⋅ dTh) ⋅ ϕ - 1e-10*(dψh ⋅ ϕ)
#      ) * dΩ

# b((uh, Th, yh, ψh), (v, R, w, ϕ)) = a((uh, Th, yh, ψh), (v, R, w, ϕ), 1.0, wh)
# jb((uh, Th, yh, ψh), (duh, dTh, dyh, dψh), (v, R, w, ϕ)) = jac((uh, Th, yh, ψh), (duh, dTh, dyh, dψh), (v, R, w, ϕ), 1.0)
# op = FEOperator(b, jb, U, V)
# Gridap.Algebra.residual(op, zh)
# Gridap.Algebra.jacobian(op, zh)



# using LineSearches: BackTracking
# nls = NLSolver(
#   show_trace=true, method=:newton, linesearch=BackTracking())
# solver = FESolver(nls)
# uh,ψh = solve(solver,op)
# lb = -1e10*ones(Float64,num_free_dofs(V))
# ub = ones(Float64,num_free_dofs(V))

# u0 = interpolate_everywhere(x->0.9*Φ0(x), Uu).free_values
# T0 = interpolate_everywhere(x->0.2, UT).free_values
# w0 = interpolate_everywhere(x->exp(Φ0(x)-0.9*Φ0(x)), Uψ).free_values

au0(uh, v) =∫(∇(uh) ⋅ ∇(v) + uh ⋅ v) * dΩ
jacu0(uh, duh, v) =∫(∇(duh) ⋅ ∇(v) + duh ⋅ v) * dΩ
J = Gridap.Algebra.jacobian( FEOperator(au0, jacu0, Uu, Vu), interpolate_everywhere(x->0.0, Uu))

u0 = interpolate_everywhere(x->0.0, Uu).free_values
T0 = interpolate_everywhere(x->1.0, UT).free_values
w0 = interpolate_everywhere(x->0.0, Uψ).free_values
zh = FEFunction(U,[u0; T0; w0]);
wh = FEFunction(Uψ,w0);
newton_its=[];
# αs = [1e-5, 0.001, 0.01, 0.02, 0.03,0.05,0.06,0.07, 0.1, 0.5, 1.0, 5.0, 1e1, 1e2, 1e3]    
# αs = [1e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 5e0, 1e1, 1e2, 1e2]
# αs = 2.0.^(-4:7)
zhs, cauchy = [zh.single_fe_functions[1].free_values[:]], []
α = 2^(-6)
for j in 1:100
    print("Considering α = $α. \n")

    b((uh, Th, ψh), (v, R, ϕ)) = a((uh, Th, ψh), (v, R, ϕ), α, wh)
    jb((uh, Th, ψh), (duh, dTh, dψh), (v, R, ϕ)) = jac((uh, Th, ψh), (duh, dTh, dψh), (v, R, ϕ), α)
    op = FEOperator(b, jb, U, V)

    nls = NLSolver(show_trace=true, method=:newton, linesearch=LineSearches.BackTracking(c_1=-1e8), ftol=1e-5, xtol=10*eps())
    solver = FESolver(nls)
    zh, its = solve!(zh,solver,op)
    its = its.result.iterations


    # zh, its = newton(op, zh, max_iter=100, damping=1, tol=1e-7, info=true);
    push!(zhs, zh.single_fe_functions[1].free_values[:])
    wh = FEFunction(Vψ, zh.single_fe_functions[3].free_values)
    push!(newton_its, its)

    d = zhs[end] - zhs[end-1]
    push!(cauchy, sqrt(d' * J * d))
    print("Cauchy norm: $(cauchy[end]).\n")

    α = 4*α
    # if its ≤ 10
    #     α = 2*α
    # # elseif its > 9
    # #     α = α/2
    # end

    if cauchy[end] < 1e-9
        break
    end
end
newton_its
cauchy

newton_its
# xx = range(0,1,100)
# membrane1 = zhs[end].single_fe_functions[1]
# Plots.plot(xx, [membrane1.(Point.(xx))],
#     linewidth=2,
#     linestyle=[:solid],
#     label="Membrane")
# # mould1 = zhs[1][end].single_fe_functions[3]
# Th = zhs[end].single_fe_functions[2]
# _mould1(x) = Φ₀(Point(x)) + φ(Point(x)) ⋅ Th(Point(x))
# xx = range(0,1,num_free_dofs(Uu))
# Plots.plot!(xx, [_mould1.(xx)],
#     linewidth=2,
#     label="Mould",
#     linestyle=[:dash])
# Plots.plot!(xx, [Φ₀(Point.(xx))],
#     linewidth=2,
#     label="Original mould",
#     linestyle=[:dashdot])


uh = zh.single_fe_functions[1]
writevtk(Ω,"membrane", cellfields=["membrane"=>uh])
Th = zh.single_fe_functions[2]
mould = interpolate_everywhere(Φ₀ + φ ⋅ Th, VT)
writevtk(Ω,"mould", cellfields=["mould"=>mould])
xx = range(0,1,101)
p = plot(xx, [uh(Point.(xx,0.5)) mould(Point.(xx,0.5)) Φ₀(Point.(xx, 0.5)) Th(Point.(xx,0.5))],
    linewidth=2,
    label=["Membrane" "Mould" "Original Mould" "Temperature"],
    linestyle=[:solid :dash],
    xlabel=L"x",
    xlabelfontsize=20, xtickfontsize=12,ytickfontsize=12,
)
Plots.savefig("thermoforming-slice.pdf")

xx = range(0,1,11)
Plots.gr_cbar_offsets[] = (-0.05,-0.01)
Plots.gr_cbar_width[] = 0.03
default(; fontfamily="Times Roman")
p = surface(xx, xx, (x, y) -> uh(Point.(x,y)), 
    color=:diverging, #:vik,
    xlabel=L"x_1", ylabel=L"x_2", zlabel=L"u(x_1,x_2)",
    # camera=(30,-30),
    title="Membrane  "*L"u",
    margin=(-6, :mm),
    cbar_title=L"u",
    colorbar_titlefontsize=20,
    # zlim=[0,1.25],
)
Plots.savefig("membrane.pdf")

xx = range(0,1,101)
Plots.gr_cbar_offsets[] = (-0.05,-0.01)
Plots.gr_cbar_width[] = 0.03
p = surface(xx, xx, (x, y) -> mould(Point.(x,y)), 
    color=:diverging, #:vik,
    xlabel=L"x_1", ylabel=L"x_2", zlabel=L"(\Phi_0 + \varphi T)(x,y)",
    # camera=(30,-30),
    title="Mould  "*L"\Phi_0 + \varphi T",
    margin=(-6, :mm),
    # zlim=[0,1.25],
)