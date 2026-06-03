using JuMP
using HiGHS

# --- Parameters ---
N = 4
alpha = 0.5
Pi = [1.0, 2.0, 3.0, 4.0]
alpha_N = alpha * N
cap = 1.0 / alpha_N

# ==========================================
# 1. PRIMAL CVaR MODEL
# ==========================================
primal_model = Model(HiGHS.Optimizer)
set_silent(primal_model)

@variable(primal_model, z)
@variable(primal_model, delta[1:N] >= 0)

@objective(primal_model, Max, z - (1.0 / alpha_N) * sum(delta[w] for w in 1:N))
@constraint(primal_model, primal_con[w=1:N], delta[w] >= z - Pi[w])

optimize!(primal_model)

z_star = value(z)
delta_star = value.(delta)
primal_obj = objective_value(primal_model)

println("=== PRIMAL RESULTS ===")
println("Optimal Objective : ", primal_obj)
println("Optimal z* : ", z_star)
println("Optimal delta* : ", delta_star)

# ==========================================
# 2. DUAL CVaR MODEL
# ==========================================
dual_model = Model(HiGHS.Optimizer)
set_silent(dual_model)

# Using beta to match the prompt's β⋆ notation
@variable(dual_model, 0 <= beta[1:N] <= cap)
@objective(dual_model, Min, sum(Pi[w] * beta[w] for w in 1:N))
@constraint(dual_model, mass_balance, sum(beta[w] for w in 1:N) == 1.0)

optimize!(dual_model)

beta_star = value.(beta)
dual_obj = objective_value(dual_model)

println("\n=== DUAL RESULTS ===")
println("Optimal Objective : ", dual_obj)
println("Optimal beta* : ", beta_star)

# ==========================================
# 3. VERIFICATION CHECKS
# ==========================================
println("\n=== CONFIRMATIONS ===")

# Check 1: Objectives agree to numerical tolerance
obj_match = isapprox(primal_obj, dual_obj, atol=1e-5)
println("1. Primal and Dual objectives agree? : ", obj_match)

# Check 2: Dual mass sums to one
mass_sum = sum(beta_star)
mass_match = isapprox(mass_sum, 1.0, atol=1e-5)
println("2. Dual mass sums to exactly 1?      : ", mass_match)

# Check 3: Cap is active on exactly the two lower-tail scenarios
println("3. Per-scenario caps (Cap = $cap):")
for w in 1:N
    is_active = isapprox(beta_star[w], cap, atol=1e-5)
    println("   Scenario $w (Pi = $(Pi[w])): beta = $(round(beta_star[w], digits=4)) -> Cap Active? $is_active")
end