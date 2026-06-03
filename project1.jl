# #Project 1 Full Submission (responsible for which questions?)
# using JuMP, HiGHS, Plots, ForwardDiff

# #Build an oracle that returns the function balue f(x) and the subgradient g defined as the derivative of f(x)
# function oracle(f,x)
#     f_val = f(x)
#     g_val = ForwardDiff.gradient(f,x)
#     return f_val, g_val
# end

# #track the upper bound, lower bound, and gap defined as the difference between the two (up to a prescribed tolerance (10^-4?))
# #Problems 1 through 3 (need to present subgradient method side by side to the kelley cutting plane method)

# #how do I choose the step size alpha?

# #n=1 #number of variables 
# xUb = 10*ones(n) 
# XLb = -5*ones(n)
# #defined function and its gradient
# function fg(x::Vector{Float64})

#     return (0.5/length(x))*x'x, (1/length(x)) * x #x'x is x transpose x
# end

# MaxIter = 1000
# x1 = XLb # initial point
# xbest = x1 #best point
# LB = [-1e6] #Lower bound 
# tol = 1e-4

# f_, g1 = fg(x1) #upper bound and gradiant at the initial point
# UB = [f_] #upper bound
# F = []
# G = []
# X = []
# push!(F, f_)
# push!(G, g1)
# push!(X, x1)
# k = 1 #count the iterations
# #print lower bound, upper bound and gap at each iteration
# println("Iter: ", k, " LB: ", LB[k], " UB: ", UB[k], " Gap: ", UB[k] - LB[k])
# #================================#
# #Initialization===========#
# #=====================#
# x1 = XLb
# xbest = x1


# model = Model(HiGHS.Optimizer)
# set_silent(model)
# @variable(model, xLb[i]<= x[i=1:n] <= xUb[i])
# @variable(model, theta)
# @objective(model, Min, theta)
# @constraint(model, theta>= F[1]+G[1]'*(x .- X[1])) #add the first cut
# #want to minimize the last dimension 

# #================================#
# #End Initialization===========#
# #=====================#
import Pkg;
Pkg.add("ForwardDiff") # download Forwarddiff
using JuMP, HiGHS, Plots, ForwardDiff, Printf

# ==========================================
# 1. ORACLE AND PROBLEM DEFINITION (P1)
# ==========================================

# First-order oracle using ForwardDiff
function oracle(f, x::Vector{Float64})
    f_val = f(x)
    g_val = ForwardDiff.gradient(f, x)
    return f_val, g_val
end

# P1 Definition
f_P1(x::Vector{Float64}) = x[1]^2  
x_lb = [-0.5]
x_ub = [1.0]
x_init = [1.0] # Starting point

tol = 1e-4
max_iter = 1000

# ==========================================
# 2. KELLEY'S CUTTING-PLANE METHOD
# ==========================================
function run_kelley(f, x0, lb, ub, max_iter, tol)
    n = length(x0)
    x_k = copy(x0)
    
    UB_hist, LB_hist, gap_hist = Float64[], Float64[], Float64[]
    X_hist = Vector{Float64}[]
    best_UB = Inf
    
    # Initialize the model
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, lb[i] <= x[i=1:n] <= ub[i])
    @variable(model, theta >= -1e6) 
    @objective(model, Min, theta)
    
    cpu_time = @elapsed begin
        for k in 1:max_iter
            f_val, g_val = oracle(f, x_k)
            push!(X_hist, copy(x_k))
            
            # Update Upper Bound
            best_UB = min(best_UB, f_val)
            push!(UB_hist, best_UB)
            
            # Add the new cutting plane to the model
            @constraint(model, theta >= f_val + sum(g_val[i] * (x[i] - x_k[i]) for i in 1:n))
            
            # Solve LP
            optimize!(model)

            # Update Lower Bound
            lb_val = termination_status(model) == MOI.OPTIMAL ? value(theta) : best_UB
            push!(LB_hist, lb_val)
            
            gap = best_UB - lb_val
            push!(gap_hist, gap)
            
            # Terminate if gap is within tolerance
            if gap <= tol
                break
            end
            
            # Next point is the optimum of the Master Problem
            x_k = value.(x)
        end
    end
    
    return X_hist, UB_hist, LB_hist, gap_hist, cpu_time
end

# ==========================================
# 3. PROJECTED SUBGRADIENT METHOD
# ==========================================
function run_subgradient(f, x0, lb, ub, max_iter, tol)
    n = length(x0)
    x_k = copy(x0)
    
    UB_hist, LB_hist, gap_hist = Float64[], Float64[], Float64[]
    X_hist = Vector{Float64}[]
    
    best_UB = Inf
    best_LB = -Inf
    
    cpu_time = @elapsed begin
        for k in 1:max_iter
            f_val, g_val = oracle(f, x_k)
            push!(X_hist, copy(x_k))
            
            # Update Upper Bound
            best_UB = min(best_UB, f_val)
            push!(UB_hist, best_UB)
            
            # Calculate a Lower Bound for the subgradient method using the current cut over the box
            # Min of (f(x_k) + g^T(x - x_k)) occurs when we push x to the bounds opposite to the gradient sign
            cut_min = f_val
            for i in 1:n
                extreme_x = g_val[i] > 0 ? lb[i] : ub[i]
                cut_min += g_val[i] * (extreme_x - x_k[i])
            end
            best_LB = max(best_LB, cut_min)
            push!(LB_hist, best_LB)
            
            gap = best_UB - best_LB
            push!(gap_hist, gap)
            
            if gap <= tol
                break
            end
            
            # Subgradient Step Size: alpha = 1 / sqrt(k)
            alpha = 1.0 / sqrt(k)
            
            # Update x and Project back onto the box [lb, ub]
            for i in 1:n
                x_raw = x_k[i] - alpha * g_val[i]
                x_k[i] = clamp(x_raw, lb[i], ub[i])
            end
        end
    end
    
    return X_hist, UB_hist, LB_hist, gap_hist, cpu_time
end

# ==========================================
# 4. RUN ALGORITHMS AND GENERATE OUTPUTS
# ==========================================

# Run Kelley
X_kel, UB_kel, LB_kel, gap_kel, time_kel = run_kelley(f_P1, x_init, x_lb, x_ub, max_iter, tol)

# Run Subgradient
X_sub, UB_sub, LB_sub, gap_sub, time_sub = run_subgradient(f_P1, x_init, x_lb, x_ub, max_iter, tol)

# --- Output 1: Summary Table ---
println("\n=== SUMMARY TABLE FOR P1 ===")
@printf("%-15s | %-12s | %-12s | %-10s | %-10s\n", "Method", "f(x*)", "Final Gap", "Iterations", "CPU Time (s)")
println("-" * 70)
@printf("%-15s | %-12.6f | %-12.6f | %-10d | %-10.6f\n", "Kelley", UB_kel[end], gap_kel[end], length(gap_kel), time_kel)
@printf("%-15s | %-12.6f | %-12.6f | %-10d | %-10.6f\n", "Subgradient", UB_sub[end], gap_sub[end], length(gap_sub), time_sub)

# --- Output 2: Convergence Plots ---
p1 = plot(1:length(gap_kel), [UB_kel, LB_kel, gap_kel], 
    label=["UB" "LB" "Gap"], title="Kelley Convergence", xlabel="Iteration", ylabel="Value", lw=2)

p2 = plot(1:length(gap_sub), [UB_sub, LB_sub, gap_sub], 
    label=["UB" "LB" "Gap"], title="Subgradient Convergence", xlabel="Iteration", ylabel="Value", lw=2)

conv_plot = plot(p1, p2, layout=(1,2), size=(900, 400))
display(conv_plot)

# --- Output 3: 1D Trial Points Plot ---
# Generate smooth curve data
x_range = range(-0.5, 1.0, length=100)
y_range = [f_P1([x]) for x in x_range]

# Extract single coordinates from the history arrays
x_kel_pts = [x[1] for x in X_kel]
y_kel_pts = [f_P1(x) for x in X_kel]

x_sub_pts = [x[1] for x in X_sub]
y_sub_pts = [f_P1(x) for x in X_sub]

p3 = plot(x_range, y_range, label="f(x) = x^2", title="Kelley Trial Points", lw=2, legend=:top)
scatter!(p3, x_kel_pts, y_kel_pts, label="Evaluated Points", color=:red, msc=:red, markersize=4)

p4 = plot(x_range, y_range, label="f(x) = x^2", title="Subgradient Trial Points", lw=2, legend=:top)
scatter!(p4, x_sub_pts, y_sub_pts, label="Evaluated Points", color=:blue, msc=:blue, markersize=4)

trial_plot = plot(p3, p4, layout=(1,2), size=(900, 400))
display(trial_plot)


