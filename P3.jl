import Pkg
Pkg.add("XLSX")
Pkg.add("JuMP")
Pkg.add("HiGHS")

using XLSX
using JuMP
using HiGHS

# --- 1. Load the Exact Matrices from Excel ---
file_path = "'/Users/vicky/Downloads/Risk-Averse Optimal Renewable Portfolio - Multicontract - 2026.xlsm'"

# Extract Wind GSF (12 months x 2000 scenarios)
# Column B to BXY covers exactly 2000 columns.
gsf_wind_raw = XLSX.readdata(file_path, "WindSimulation", "B2:BXY13")
gsf_wind = convert(Matrix{Float64}, gsf_wind_raw)

# Extract Northeast Spot Prices (12 months x 2000 scenarios)
# Assuming SE prices are rows 2-13, and NE prices start at row 16.
pi_NE_raw = XLSX.readdata(file_path, "SpotPriceSimulation", "B16:BXY27")
pi_NE = convert(Matrix{Float64}, pi_NE_raw)

println("Successfully loaded matrices!")
println("Wind GSF size: ", size(gsf_wind))
println("NE Spot Price size: ", size(pi_NE))

# --- 2. P4 Portfolio Optimization with Call Options ---

# Constants & Parameters
N = 2000
T = 12
hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]

P_sell = 140.0
P_wind = 100.0
Q_wind_max = 11.41
alpha = 0.05
lambda = 0.5
Strike_K = 150.0 

# Pre-calculate option payout matrix (keeps LP linear)
option_payout = max.(0.0, pi_NE .- Strike_K)

function solve_portfolio_with_options(Premium_C)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    # Decision Variables
    @variable(model, 0 <= Q_sell <= 100)
    @variable(model, 0 <= Q_wind <= Q_wind_max)
    @variable(model, 0 <= Q_call <= 100)  # The new hedging variable
    
    # CVaR Variables
    @variable(model, z)
    @variable(model, delta[1:N] >= 0)
    @variable(model, profit[1:N])
    
    # Constraints: Scenario Profits
    for w in 1:N
        annual_profit = @expression(model, sum(
            hours[t] * (
                P_sell * Q_sell - P_wind * Q_wind + 
                pi_NE[t, w] * (gsf_wind[t, w] * Q_wind - Q_sell) +
                Q_call * option_payout[t, w] - Q_call * Premium_C
            ) for t in 1:T
        ))
        @constraint(model, profit[w] == annual_profit)
        
        # CVaR tail constraints
        @constraint(model, delta[w] >= z - profit[w])
    end
    
    # Expected Profit & CVaR Formulation
    @variable(model, exp_profit)
    @constraint(model, exp_profit == sum(profit[w] for w in 1:N) / N)
    
    @variable(model, cvar_val)
    @constraint(model, cvar_val == z - (1 / (alpha * N)) * sum(delta[w] for w in 1:N))
    
    # Objective: Maximize Certainty Equivalent (Risk-Adjusted Return)
    @objective(model, Max, lambda * cvar_val + (1 - lambda) * exp_profit)
    
    optimize!(model)
    
    return value(Q_sell), value(Q_wind), value(Q_call), value(exp_profit), value(cvar_val)
end

# --- 3. The Numerical Study (Sweeping the Premium) ---
premium_grid = 1.0:2.0:30.0 
results = []

println("\nPremium | Q_sell | Q_wind | Q_call | Exp_Profit | CVaR")
println("-"^60)

for C in premium_grid
    q_s, q_w, q_c, ep, cv = solve_portfolio_with_options(C)
    push!(results, (C, q_s, q_w, q_c, ep, cv))
    
    # Formatting output for a clean table
    println("$(lpad(C, 7)) | $(round(q_s, digits=2)) | $(round(q_w, digits=2)) | $(round(q_c, digits=2)) | $(round(ep, digits=0)) | $(round(cv, digits=0))")
end