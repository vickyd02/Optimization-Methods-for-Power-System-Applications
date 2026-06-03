import Pkg; 
Pkg.add("JuMP")
Pkg.add("HiGHS")
using JuMP, HiGHS

# ── Data ──────────────────────────────────────────────────────────────────────
p = [100.0, 150.0, 500.0]   # offer prices $/MWh
Q = [100.0,  50.0, 120.0]   # capacities MWh
d = 120.0                    # demand MWh
n = length(p)

# ── Primal ────────────────────────────────────────────────────────────────────
primal = Model(HiGHS.Optimizer)
set_silent(primal)

@variable(primal, g[i=1:n] >= 0)
@objective(primal, Min, sum(p[i]*g[i] for i in 1:n))
@constraint(primal, energy_balance, sum(g[i] for i in 1:n) == d)
@constraint(primal, capacity[i=1:n], g[i] <= Q[i])

optimize!(primal)

g_star = value.(g)
C_star = objective_value(primal)
pi_star = dual(energy_balance)
beta_star = -dual.(capacity)   # negate: JuMP returns ≤0 for <= constraints in minimization

println("=" ^ 50)
println("PRIMAL SOLUTION")
println("=" ^ 50)
println("g*       = ", g_star)
println("C*       = ", C_star, " \$")
println("π*       = ", pi_star, " \$/MWh")
println("β*       = ", beta_star, " \$/MWh")

# ── Dual ──────────────────────────────────────────────────────────────────────
dual_model = Model(HiGHS.Optimizer)
set_silent(dual_model)

@variable(dual_model, pi_d)                      # free
@variable(dual_model, beta_d[i=1:n] >= 0)        # non-negative

@objective(dual_model, Max,
    d * pi_d - sum(Q[i]*beta_d[i] for i in 1:n))

@constraint(dual_model, dual_con[i=1:n],
    pi_d - beta_d[i] <= p[i])

optimize!(dual_model)

pi_d_star    = value(pi_d)
beta_d_star  = value.(beta_d)
dual_obj     = objective_value(dual_model)

println()
println("=" ^ 50)
println("DUAL SOLUTION")
println("=" ^ 50)
println("π*       = ", pi_d_star,   " \$/MWh")
println("β*       = ", beta_d_star, " \$/MWh")
println("Dual obj = ", dual_obj,    " \$")

# ── Verification ──────────────────────────────────────────────────────────────
println()
println("=" ^ 50)
println("VERIFICATION")
println("=" ^ 50)
println("Primal obj C*             = ", round(C_star,    digits=6))
println("Dual obj                  = ", round(dual_obj,  digits=6))
println("Strong duality holds?       ",
    abs(C_star - dual_obj) < 1e-6 ? "YES ✓" : "NO ✗")
println()
println("Inframarginal rent Σβ*g*  = ",
    round(sum(beta_star[i]*g_star[i] for i in 1:n), digits=6))
println("π*d - C*                  = ",
    round(pi_star*d - C_star, digits=6))
println("Rents equal surplus?        ",
    abs(sum(beta_star[i]*g_star[i] for i in 1:n) -
        (pi_star*d - C_star)) < 1e-6 ? "YES ✓" : "NO ✗")



using JuMP, HiGHS

# ── Data ──────────────────────────────────────────────────────────────────────
p  = [100.0, 150.0, 500.0]          # energy offer prices $/MWh
Q  = [100.0,  50.0, 120.0]          # capacities MWh
cr = 0.2 .* p                        # reserve costs = 20% of energy offer
d  = 120.0                           # demand MWh
n  = length(p)                       # number of generators
K  = 1                               # security parameter (n-1)
R  = Q                               # reserve upper bound = capacity

# ─────────────────────────────────────────────────────────────────────────────
# FORMULATION 1: MONOLITHIC
# Explicitly enumerate all n contingency constraints (one per generator outage)
# ─────────────────────────────────────────────────────────────────────────────
mono = Model(HiGHS.Optimizer)
set_silent(mono)

@variable(mono, 0 <= g_m[i=1:n] <= Q[i])
@variable(mono, 0 <= r_m[i=1:n] <= R[i])

@objective(mono, Min,
    sum(p[i]*g_m[i] + cr[i]*r_m[i] for i in 1:n))

# Energy balance
@constraint(mono, energy_balance_m,
    sum(g_m[i] for i in 1:n) == d)

# Capacity: dispatch + reserve cannot exceed capacity
@constraint(mono, cap_m[i=1:n],
    g_m[i] + r_m[i] <= Q[i])

# Contingency constraints: for each outage k, surviving units must cover demand
# Generator k fails: sum over i != k of (g_i + r_i) >= d
@constraint(mono, contingency_m[k=1:n],
    sum(g_m[i] + r_m[i] for i in 1:n if i != k) >= d)

optimize!(mono)

g_mono   = value.(g_m)
r_mono   = value.(r_m)
obj_mono = objective_value(mono)

println("=" ^ 55)
println("MONOLITHIC FORMULATION")
println("=" ^ 55)
println("Optimal cost  = \$", round(obj_mono, digits=4))
println("Dispatch g*   = ", round.(g_mono, digits=4), " MWh")
println("Reserve  r*   = ", round.(r_mono, digits=4), " MWh")
println("g* + r*       = ", round.(g_mono .+ r_mono, digits=4), " MWh")

# ─────────────────────────────────────────────────────────────────────────────
# FORMULATION 2: ROBUST COUNTERPART (via duality)
# Replace all contingency constraints with dual absorption
# Inner LP:  min  sum_i a_i(g_i + r_i)
#            s.t. sum_i a_i >= n - K      [dual: y >= 0]
#                 0 <= a_i <= 1           [dual: z_i >= 0]
# Dual of inner LP:
#            max  (n-K)*y - sum_i z_i
#            s.t. y - z_i <= g_i + r_i   for all i
#                 z_i >= 0, y >= 0
# Robust security constraint: (n-K)*y - sum_i z_i >= d
# ─────────────────────────────────────────────────────────────────────────────
rob = Model(HiGHS.Optimizer)
set_silent(rob)

@variable(rob, 0 <= g_r[i=1:n] <= Q[i])
@variable(rob, 0 <= r_r[i=1:n] <= R[i])
@variable(rob, y >= 0)                   # dual of budget constraint
@variable(rob, z[i=1:n] >= 0)           # dual of per-generator bound

@objective(rob, Min,
    sum(p[i]*g_r[i] + cr[i]*r_r[i] for i in 1:n))

# Energy balance
@constraint(rob, energy_balance_r,
    sum(g_r[i] for i in 1:n) == d)

# Capacity: dispatch + reserve cannot exceed capacity
@constraint(rob, cap_r[i=1:n],
    g_r[i] + r_r[i] <= Q[i])

# Robust security constraint: dual objective >= d
@constraint(rob, robust_security,
    (n - K)*y - sum(z[i] for i in 1:n) >= d)

# Dual feasibility constraints: y - z_i <= g_i + r_i
@constraint(rob, dual_feas[i=1:n],
    y - z[i] <= g_r[i] + r_r[i])

optimize!(rob)

g_rob   = value.(g_r)
r_rob   = value.(r_r)
y_star  = value(y)
z_star  = value.(z)
obj_rob = objective_value(rob)

println()
println("=" ^ 55)
println("ROBUST COUNTERPART")
println("=" ^ 55)
println("Optimal cost  = \$", round(obj_rob,  digits=4))
println("Dispatch g*   = ", round.(g_rob,   digits=4), " MWh")
println("Reserve  r*   = ", round.(r_rob,   digits=4), " MWh")
println("g* + r*       = ", round.(g_rob .+ r_rob, digits=4), " MWh")
println("y*            = ", round(y_star,   digits=4))
println("z*            = ", round.(z_star,  digits=4))

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
println()
println("=" ^ 55)
println("VERIFICATION")
println("=" ^ 55)
println("Monolithic obj   = \$", round(obj_mono, digits=6))
println("Robust obj       = \$", round(obj_rob,  digits=6))
println("Objectives match?  ",
    abs(obj_mono - obj_rob) < 1e-4 ? "YES ✓" : "NO ✗")
println()
println("Dispatch match?    ",
    all(abs.(g_mono .- g_rob) .< 1e-4) ? "YES ✓" : "NO ✗")
println("Reserve match?     ",
    all(abs.(r_mono .- r_rob) .< 1e-4) ? "YES ✓" : "NO ✗")
println()

# Check that each contingency is actually satisfied by the robust solution
println("Contingency feasibility check (robust solution):")
for k in 1:n
    available = sum(g_rob[i] + r_rob[i] for i in 1:n if i != k)
    println("  Lose generator $k: available = ",
        round(available, digits=4),
        " MWh  >= d = $d ?  ",
        available >= d - 1e-6 ? "YES ✓" : "NO ✗")
end

# Check inner LP worst case matches dual certificate
println()
println("Dual certificate check:")
println("  (n-K)*y* - Σz* = ",
    round((n-K)*y_star - sum(z_star), digits=4),
    "  >= d = $d ?  ",
    (n-K)*y_star - sum(z_star) >= d - 1e-6 ? "YES ✓" : "NO ✗")

# =============================================================================
# Two-Stage Stochastic Dispatch with Wind Uncertainty
# =============================================================================
# Stage 1 (here-and-now): choose thermal dispatch gᵢ and reserves rᵢ
# Stage 2 (wait-and-see): for each wind scenario ω, adjust dispatch,
#                          curtail wind, or shed load
#
# Solved via extensive form (all scenarios simultaneously).
# Dual variable λω (real-time LMP) is recovered after solving.
# =============================================================================

using JuMP, HiGHS, Statistics, Printf, Random

# =============================================================================
# USER-ADJUSTABLE PARAMETERS — tweak anything here
# =============================================================================

# --- Thermal generators (i = 1, 2, 3) ----------------------------------------
p  = [100.0, 150.0, 500.0]   # [$/MWh] offer prices
Q  = [100.0,  50.0, 120.0]   # [MWh]   capacity blocks

# --- Wind generator -----------------------------------------------------------
p_wind   = 0.0                # [$/MWh] wind marginal cost (zero)
Q_wind   = 50.0               # [MWh]   installed wind capacity

# --- Demand -------------------------------------------------------------------
d = 120.0                     # [MWh]   inelastic demand (deterministic)

# --- Reserve cost -------------------------------------------------------------
reserve_rate = 0.20           # fraction of pᵢ charged per MWh of held reserve

# --- Recourse costs -----------------------------------------------------------
c_curt = 10.0                 # [$/MWh] cost of curtailing wind
c_voll = 5_000.0              # [$/MWh] value of lost load (unserved energy)

# --- Scenario generation ------------------------------------------------------
S          = 1_000            # number of wind scenarios
wind_seed  = 42               # RNG seed (change for different realisations)

# Wind distribution: Beta(α, β) scaled to [0, Q_wind]
# Beta(2,2) gives a symmetric bell; Beta(1,5) skews low; Beta(5,1) skews high
wind_alpha = 2.0
wind_beta  = 5.0              # skewed-low: wind is often disappointing

# =============================================================================
# SCENARIO GENERATION
# =============================================================================

Random.seed!(wind_seed)

# Draw wind realisations from Beta(α,β) scaled to [0, Q_wind]
function draw_wind_scenarios(S, Q_wind, α, β)
    # Use the relation: Beta sample via two Gamma variates
    w = zeros(S)
    for s in 1:S
        x = rand(Gamma(α, 1.0))
        y = rand(Gamma(β, 1.0))
        w[s] = Q_wind * x / (x + y)
    end
    return w
end

using Distributions   # for Gamma distribution
w_tilde = draw_wind_scenarios(S, Q_wind, wind_alpha, wind_beta)

prob = fill(1.0/S, S)   # equal probability weights

# =============================================================================
# BUILD AND SOLVE THE EXTENSIVE FORM
# =============================================================================

n = length(p)   # number of thermal generators

model = Model(HiGHS.Optimizer)
set_silent(model)

# --- Stage 1 variables -------------------------------------------------------
@variable(model, 0 <= g[i=1:n] <= Q[i])          # dispatch
@variable(model, 0 <= r[i=1:n])                   # reserves (upper bound below)
@constraint(model, [i=1:n], g[i] + r[i] <= Q[i]) # capacity shared between g and r

# --- Stage 2 variables (one set per scenario) --------------------------------
@variable(model, dg[i=1:n, s=1:S])               # adjustment (can be negative)
@variable(model, 0 <= curt[s=1:S] <= Q_wind)     # wind curtailment
@variable(model, 0 <= unsrv[s=1:S])              # unserved load

# --- Stage 2 bounds: adjustment within [-gᵢ, rᵢ] ----------------------------
@constraint(model, adj_up[i=1:n, s=1:S],   dg[i,s] <=  r[i])
@constraint(model, adj_dn[i=1:n, s=1:S],  -dg[i,s] <=  g[i])

# --- Stage 2 power balance (one per scenario) --------------------------------
# This constraint's dual is λω
@constraint(model, balance[s=1:S],
    sum(g[i] + dg[i,s] for i in 1:n) + w_tilde[s] - curt[s] - unsrv[s] == d
)

# --- Objective: Stage 1 costs + expected Stage 2 costs -----------------------
@objective(model, Min,
    sum(p[i]*g[i] for i in 1:n)                              # dispatch cost
  + sum(reserve_rate * p[i] * r[i] for i in 1:n)            # reserve holding cost
  + sum(prob[s] * (
        sum(p[i]*dg[i,s] for i in 1:n)                      # redispatch cost
      + c_curt  * curt[s]                                    # curtailment cost
      + c_voll  * unsrv[s]                                   # VOLL
    ) for s in 1:S)
)

optimize!(model)

# =============================================================================
# EXTRACT RESULTS
# =============================================================================

status = termination_status(model)
obj    = objective_value(model)

g_val  = value.(g)
r_val  = value.(r)
dg_val = value.(dg)
curt_val  = value.(curt)
unsrv_val = value.(unsrv)

# Dual variables: λω = LMP in each scenario
lambda = dual.(balance)   # vector length S (positive = $/MWh scarcity rent)

# Per-scenario realised cost breakdown
stage2_cost = [
    sum(p[i]*dg_val[i,s] for i in 1:n) + c_curt*curt_val[s] + c_voll*unsrv_val[s]
    for s in 1:S
]

stage1_dispatch_cost = sum(p[i]*g_val[i] for i in 1:n)
stage1_reserve_cost  = sum(reserve_rate * p[i] * r_val[i] for i in 1:n)
expected_stage2_cost = sum(prob[s]*stage2_cost[s] for s in 1:S)

# =============================================================================
# PRINT RESULTS
# =============================================================================

println("=" ^ 60)
println("  TWO-STAGE STOCHASTIC DISPATCH — RESULTS")
println("=" ^ 60)
println("Solver status : ", status)
println()

println("── Stage 1 Decisions ──────────────────────────────────")
println(@sprintf("  %-12s  %-8s  %-8s  %-8s  %-8s", "Generator", "p (\$/MWh)", "Q (MWh)", "g* (MWh)", "r* (MWh)"))
for i in 1:n
    println(@sprintf("  %-12s  %-8.1f  %-8.1f  %-8.2f  %-8.2f",
        "G$i", p[i], Q[i], g_val[i], r_val[i]))
end
println(@sprintf("  %-12s  %-8.1f  %-8.1f  %-8s  %-8s",
    "Wind", p_wind, Q_wind, "stochastic", "—"))
println()

println("── Cost Breakdown ─────────────────────────────────────")
println(@sprintf("  Stage 1 dispatch cost    : \$%10.2f", stage1_dispatch_cost))
println(@sprintf("  Stage 1 reserve cost     : \$%10.2f", stage1_reserve_cost))
println(@sprintf("  Expected Stage 2 cost    : \$%10.2f", expected_stage2_cost))
println(@sprintf("  ─────────────────────────────────────────"))
println(@sprintf("  Total expected cost      : \$%10.2f", obj))
println()

println("── Dual Variables (Real-Time LMPs λω) ────────────────")
println(@sprintf("  Mean λω                  : \$%8.2f / MWh", mean(lambda)))
println(@sprintf("  Std  λω                  : \$%8.2f / MWh", std(lambda)))
println(@sprintf("  Min  λω                  : \$%8.2f / MWh", minimum(lambda)))
println(@sprintf("  Max  λω                  : \$%8.2f / MWh", maximum(lambda)))
println()

# Fraction of scenarios with unserved load or curtailment
n_unsrv = count(unsrv_val .> 1e-4)
n_curt  = count(curt_val  .> 1e-4)
println("── Scenario Statistics ────────────────────────────────")
println(@sprintf("  Scenarios with unserved load : %d / %d  (%.1f%%)",
    n_unsrv, S, 100*n_unsrv/S))
println(@sprintf("  Scenarios with curtailment   : %d / %d  (%.1f%%)",
    n_curt,  S, 100*n_curt/S))
println(@sprintf("  Mean wind realisation        : %.2f MWh  (capacity: %.1f MWh)",
    mean(w_tilde), Q_wind))
println(@sprintf("  Mean unserved load           : %.4f MWh/scenario", mean(unsrv_val)))
println(@sprintf("  Mean curtailment             : %.4f MWh/scenario", mean(curt_val)))
println()

println("── Reserve Economics ──────────────────────────────────")
println("  Endogenous reserve price E[λω] vs reserve holding cost r·pᵢ:")
for i in 1:n
    hold_cost = reserve_rate * p[i]
    println(@sprintf("    G%d: holding cost = \$%.2f/MWh,  E[λω] = \$%.2f/MWh  → reserve %s",
        i, hold_cost, mean(lambda),
        r_val[i] > 1e-4 ? @sprintf("procured (%.2f MWh)", r_val[i]) : "not procured"))
end
println()
println("  Interpretation: generator deploys reserve when λω > pᵢ.")
println("  E[λω] = ", @sprintf("\$%.2f", mean(lambda)), " is the endogenous capacity price.")
println("=" ^ 60)

# =============================================================================
# Two-Stage Stochastic Dispatch with Wind Uncertainty
# =============================================================================
# Stage 1 (here-and-now): choose thermal dispatch gᵢ and reserves rᵢ
# Stage 2 (wait-and-see): for each wind scenario ω, adjust dispatch,
#                          curtail wind, or shed load
#
# Solved via extensive form (all scenarios simultaneously).
# Dual variable λω (real-time LMP) is recovered after solving.
# =============================================================================

using JuMP, HiGHS, Statistics, Printf, Random

# =============================================================================
# USER-ADJUSTABLE PARAMETERS — tweak anything here
# =============================================================================

# --- Thermal generators (i = 1, 2, 3) ----------------------------------------
p  = [100.0, 150.0, 500.0]   # [$/MWh] offer prices
Q  = [100.0,  50.0, 120.0]   # [MWh]   capacity blocks

# --- Wind generator -----------------------------------------------------------
p_wind   = 0.0                # [$/MWh] wind marginal cost (zero)
Q_wind   = 50.0               # [MWh]   installed wind capacity

# --- Demand -------------------------------------------------------------------
d = 250.0                     # [MWh]   inelastic demand (deterministic)

# --- Reserve cost -------------------------------------------------------------
reserve_rate = 0.20           # fraction of pᵢ charged per MWh of held reserve

# --- Recourse costs -----------------------------------------------------------
c_curt = 100.0                 # [$/MWh] cost of curtailing wind
c_voll = 100000.0              # [$/MWh] value of lost load (unserved energy)

# --- Scenario generation ------------------------------------------------------
S          = 1000            # number of wind scenarios
wind_seed  = 42               # RNG seed (change for different realisations)

# Wind distribution: Beta(α, β) scaled to [0, Q_wind]
# Beta(2,2) gives a symmetric bell; Beta(1,5) skews low; Beta(5,1) skews high
wind_alpha = 2.0
wind_beta  = 5.0              # skewed-low: wind is often disappointing

# =============================================================================
# SCENARIO GENERATION
# =============================================================================

Random.seed!(wind_seed)

# Draw wind realisations from Beta(α,β) scaled to [0, Q_wind]
function draw_wind_scenarios(S, Q_wind, α, β)
    # Use the relation: Beta sample via two Gamma variates
    w = zeros(S)
    for s in 1:S
        x = rand(Gamma(α, 1.0))
        y = rand(Gamma(β, 1.0))
        w[s] = Q_wind * x / (x + y)
    end
    return w
end

using Distributions   # for Gamma distribution
w_tilde = draw_wind_scenarios(S, Q_wind, wind_alpha, wind_beta)

prob = fill(1.0/S, S)   # equal probability weights

# =============================================================================
# MODEL - REFORMULATED
# =============================================================================

n = length(p)

model = Model(HiGHS.Optimizer)
set_silent(model)

# --- Stage 1 variables -------------------------------------------------------
@variable(model, g[i=1:n] >= 0)     # day-ahead scheduled dispatch
@variable(model, r[i=1:n] >= 0)     # reserved capacity

@constraint(model, [i=1:n], g[i] <= Q[i])
@constraint(model, [i=1:n], g[i] + r[i] <= Q[i])   # capacity shared

# --- Stage 2 variables -------------------------------------------------------
# Instead of adjustments, use TOTAL real-time dispatch
@variable(model, g_rt[i=1:n, s=1:S] >= 0)   # real-time dispatch in scenario s

# Real-time dispatch bounds
@constraint(model, [i=1:n, s=1:S], g_rt[i,s] <= Q[i])
@constraint(model, [i=1:n, s=1:S], g_rt[i,s] <= g[i] + r[i])  # can use day-ahead + reserve
@constraint(model, [i=1:n, s=1:S], g_rt[i,s] >= g[i] - g[i])  # can shut down completely (this simplifies to g_rt >= 0)

# Actually, the key constraint: real-time dispatch is limited by what was committed
# You can use up to g[i] + r[i] (what you scheduled plus what you reserved)
# You can go down to 0 (full backdown allowed for simplicity)

@variable(model, curt[s=1:S] >= 0)    # wind curtailment
@variable(model, unsrv[s=1:S] >= 0)   # unserved load

@constraint(model, [s=1:S], curt[s] <= Q_wind)

# --- Stage 2 power balance ---------------------------------------------------
@constraint(model, balance[s=1:S],
    d == sum(g_rt[i,s] for i in 1:n) + w_tilde[s] - curt[s] - unsrv[s]
)

# --- Objective ---------------------------------------------------------------
# Stage 1: pay for day-ahead dispatch + reserve holding cost
# Stage 2: pay for ACTUAL fuel consumed (g_rt), plus curtailment and VOLL
@objective(model, Min,
    sum(reserve_rate * p[i] * r[i] for i in 1:n)            # reserve holding cost
  + sum(prob[s] * (
        sum(p[i] * g_rt[i,s] for i in 1:n)                 # actual fuel cost
      + c_curt  * curt[s]
      + c_voll  * unsrv[s]
    ) for s in 1:S)
)

optimize!(model)

# =============================================================================
# EXTRACT RESULTS
# =============================================================================

status = termination_status(model)
obj    = objective_value(model)

g_val  = value.(g)
g_rt_val  = value.(g_rt)  # Extracts the optimal values of the g_rt variables
r_val  = value.(r)
dg_val = value.(dg)
curt_val  = value.(curt)
unsrv_val = value.(unsrv)

# Dual variables: λω = LMP in each scenario
lambda_raw = dual.(balance)

# Per-scenario realised cost breakdown
stage2_cost = [
    sum(p[i]*dg_val[i,s] for i in 1:n) + c_curt*curt_val[s] + c_voll*unsrv_val[s]
    for s in 1:S
]

stage1_dispatch_cost = sum(p[i]*g_val[i] for i in 1:n)
stage1_reserve_cost  = sum(reserve_rate * p[i] * r_val[i] for i in 1:n)
expected_stage2_cost = sum(prob[s]*stage2_cost[s] for s in 1:S)

# =============================================================================
# PRINT RESULTS
# =============================================================================

println("=" ^ 60)
println("  TWO-STAGE STOCHASTIC DISPATCH — RESULTS")
println("=" ^ 60)
println("Solver status : ", status)
println()

println("── Stage 1 Decisions ──────────────────────────────────")
println(@sprintf("  %-12s  %-8s  %-8s  %-8s  %-8s", "Generator", "p (\$/MWh)", "Q (MWh)", "g* (MWh)", "r* (MWh)"))
for i in 1:n
    println(@sprintf("  %-12s  %-8.1f  %-8.1f  %-8.2f  %-8.2f",
        "G$i", p[i], Q[i], g_val[i], r_val[i]))
end
println(@sprintf("  %-12s  %-8.1f  %-8.1f  %-8s  %-8s",
    "Wind", p_wind, Q_wind, "stochastic", "—"))
println()

println("── Cost Breakdown ─────────────────────────────────────")
println(@sprintf("  Stage 1 dispatch cost    : \$%10.2f", stage1_dispatch_cost))
println(@sprintf("  Stage 1 reserve cost     : \$%10.2f", stage1_reserve_cost))
println(@sprintf("  Expected Stage 2 cost    : \$%10.2f", expected_stage2_cost))
println(@sprintf("  ─────────────────────────────────────────"))
println(@sprintf("  Total expected cost      : \$%10.2f", obj))
println()

println("── Dual Variables (Real-Time LMPs λω) ────────────────")
println(@sprintf("  Mean λω                  : \$%8.2f / MWh", mean(lambda)))
println(@sprintf("  Std  λω                  : \$%8.2f / MWh", std(lambda)))
println(@sprintf("  Min  λω                  : \$%8.2f / MWh", minimum(lambda)))
println(@sprintf("  Max  λω                  : \$%8.2f / MWh", maximum(lambda)))
println()

# Fraction of scenarios with unserved load or curtailment
n_unsrv = count(unsrv_val .> 1e-4)
n_curt  = count(curt_val  .> 1e-4)
println("── Scenario Statistics ────────────────────────────────")
println(@sprintf("  Scenarios with unserved load : %d / %d  (%.1f%%)",
    n_unsrv, S, 100*n_unsrv/S))
println(@sprintf("  Scenarios with curtailment   : %d / %d  (%.1f%%)",
    n_curt,  S, 100*n_curt/S))
println(@sprintf("  Mean wind realisation        : %.2f MWh  (capacity: %.1f MWh)",
    mean(w_tilde), Q_wind))
println(@sprintf("  Mean unserved load           : %.4f MWh/scenario", mean(unsrv_val)))
println(@sprintf("  Mean curtailment             : %.4f MWh/scenario", mean(curt_val)))
println()

println("── Reserve Economics ──────────────────────────────────")
println("  Endogenous reserve price E[λω] vs reserve holding cost r·pᵢ:")
for i in 1:n
    hold_cost = reserve_rate * p[i]
    println(@sprintf("    G%d: holding cost = \$%.2f/MWh,  E[λω] = \$%.2f/MWh  → reserve %s",
        i, hold_cost, mean(lambda),
        r_val[i] > 1e-4 ? @sprintf("procured (%.2f MWh)", r_val[i]) : "not procured"))
end
println()
println("  Interpretation: generator deploys reserve when λω > pᵢ.")
println("  E[λω] = ", @sprintf("\$%.2f", mean(lambda)), " is the endogenous capacity price.")
println("=" ^ 60)

# =============================================================================
# WIND UTILIZATION DIAGNOSTICS
# =============================================================================

# Calculate actual wind used in each scenario
wind_used = [w_tilde[s] - curt_val[s] for s in 1:S]

println("\n── Wind Utilization Analysis ────────────────────────────")
println(@sprintf("  Mean wind available        : %8.2f MWh", mean(w_tilde)))
println(@sprintf("  Mean wind curtailed        : %8.2f MWh", mean(curt_val)))
println(@sprintf("  Mean wind USED             : %8.2f MWh", mean(wind_used)))
println(@sprintf("  Wind utilization rate      : %8.1f%%", 100*mean(wind_used)/mean(w_tilde)))
println()

# Show average dispatch by source
avg_thermal = [mean(g_rt_val[i,:]) for i in 1:n]
println("  Average real-time dispatch by source:")
for i in 1:n
    println(@sprintf("    G%d: %8.2f MWh  (%.1f%% of demand)",
        i, avg_thermal[i], 100*avg_thermal[i]/d))
end
println(@sprintf("    Wind: %8.2f MWh  (%.1f%% of demand)",
    mean(wind_used), 100*mean(wind_used)/d))
println(@sprintf("    Unserved: %8.2f MWh  (%.1f%% of demand)",
    mean(unsrv_val), 100*mean(unsrv_val)/d))
println()

# Check supply-demand balance
total_supply = [sum(g_rt_val[i,s] for i in 1:n) + wind_used[s] - unsrv_val[s] for s in 1:S]
println(@sprintf("  Supply-demand balance check: %.6f MWh (should be %.1f)",
    mean(total_supply), d))