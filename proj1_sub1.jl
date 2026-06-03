using JuMP, HiGHS

function solve_cutting_plane_LP(m::Int, n::Int, X_A, X_b, x_points, f_vals, g_vals)
    # m: number of points evaluated so far
    # n: dimensionality of the variable x
    # X_A, X_b: represent the domain X as a polyhedron (X_A * x <= X_b)
    # x_points: array of evaluated points (x_i)
    # f_vals: array of function values at x_i
    # g_vals: array of subgradients at x_i

    model = Model(HiGHS.Optimizer)
    
    @variable(model, x[1:n])   # Decision variable
    @variable(model, t)        # Auxiliary variable for the max of the cuts
    
    @objective(model, Min, t)  # Minimize the auxiliary variable
    
    # Constraints for the cuts
    for i in 1:m
        @constraint(model, f_vals[i] + sum(g_vals[i][j] * (x[j] - x_points[i][j]) for j in 1:n) <= t)
    end
    
    # Add constraints for the domain X
    @constraint(model, X_A * x .<= X_b)
    optimize!(model)
    
end
