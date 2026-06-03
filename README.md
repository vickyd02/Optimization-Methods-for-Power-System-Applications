# ECON 238: Optimization Methods in Power Systems
**Stanford University** | Graduate-Level Course

---

## Overview

This repo contains course material and projects from ECON 238, a graduate course at Stanford that bridges mathematical optimization theory and real-world energy economics applications. The course develops covers both the algorithmic machinery of optimization (convex analysis, cutting-plane methods, linear and mixed-integer programming) and the domain-specific modeling challenges that arise in power systems — transmission network design, renewable integration, cost allocation, and risk-aware portfolio planning.

This work is primarily done in **Julia** (using `JuMP` and `HiGHS`), with some Python used for data analysis and simulation. Projects involve formulating and solving optimization problems from scratch, interpreting economic implications of results, and communicating findings clearly.

---

## Course Objectives

The material taught us how to:

- Prove and apply fundamental properties of convex functions and convex optimization problems
- Implement cutting-plane (Kelley) methods, subgradient methods, and LP-based algorithms
- Formulate network flow and investment problems as linear programs and mixed-integer linear programs
- Apply cooperative game theory concepts — the core, the nucleolus, and excess-maximization LPs — to cost allocation problems
- Model stochastic generation profiles using copulas and simulate correlated renewable outputs
- Construct risk-averse portfolio optimization models using CVaR and stochastic programming
- Interpret the economic meaning of dual variables, binding constraints, and stability conditions

---

## Programming Environment

| Tool | Purpose |
|---|---|
| **Julia** | Primary language for all optimization projects |
| **JuMP.jl** | Algebraic modeling interface for LPs, MILPs |
| **HiGHS.jl** | Open-source LP/MILP solver backend |
| **Plots.jl / StatsPlots.jl** | Visualization |
| **Python** | Data generation, copula simulation, supplementary analysis |

---

## Course Topics

### Module 1 — Foundations of Convex Optimization

**Key concepts:**
- Definition and geometric interpretation of convex sets and convex functions
- Subgradients and subdifferential calculus
- Pointwise maximum of convex functions is convex (formally proved: if each $f_i$ is convex, then $f(x) = \max_{i \in [m]} \{f_i(x)\}$ is convex)
- Epigraph characterization of convexity

### Proof Techniques Covered
The course emphasizes proof-writing. A central early result is:

If $f_i$ is convex for all $i \in [m]$, then $f(x) = \max_{i \in [m]} f_i(x)$ is convex.

**Proof sketch:** By definition of convexity, $f_i(\alpha x_1 + (1 - \alpha)x_2) \le \alpha f_i(x_1) + (1 - \alpha)f_i(x_2)$. Taking the max over $i$ on both sides and noting that the maximum of any $f_i(x)$ is $f(x)$ by definition yields the convexity inequality for $f$.

---

### Module 2 — Cutting-Plane (Kelley) Method

The **Cutting-Plane Method** solves $\min_{x \in X} f(x)$ for convex $f$ by iteratively building a piecewise-linear lower approximation:

$$f_k(x) = \max_{i=1,\ldots,k} \left\lbrace f(x^i) + (g^i)^\top (x - x^i) \right\rbrace$$

**Algorithm:**

1. Choose $x^1 \in X$, set $k = 1$
2. Compute $f(x^k)$ and subgradient $g^k$
3. Update lower model $f_k(x)$ by adding new cutting plane
4. Solve $x^{k+1} = \arg\min_{x \in X} f_k(x)$ via LP (auxiliary variable $t$; minimize $t$ subject to $f_k(x) \leq t$)
5. Check convergence: if $f(x^{k+1}) - f_k(x^{k+1}) \leq \varepsilon$, stop
6. Otherwise set $k \leftarrow k+1$ and return to step 2

**Key monotonicity property:** $f_k(x) \leq f_{k+1}(x) \leq f(x)$ for all $x$ — each iteration tightens the lower bound without overshooting the true objective.

**Julia implementation** (`JuMP`):
```julia
function solve_cutting_plane_LP(m, n, X_A, X_b, x_points, f_vals, g_vals)
    model = Model(HiGHS.Optimizer)
    @variable(model, x[1:n])
    @variable(model, t)    # auxiliary variable for max of cuts
    @objective(model, Min, t)
    for i in 1:m
        @constraint(model, f_vals[i] + sum(g_vals[i][j] * (x[j] - x_points[i][j])
                    for j in 1:n) <= t)
    end
    @constraint(model, X_A * x .<= X_b)
    optimize!(model)
end
```

---

### Module 3 — Transmission Network Design as a Linear Program

**Problem:** Minimize the total investment cost of building transmission lines to connect $n$ renewable generators to a substation (hub node 0), subject to power flow balance and line capacity constraints.

**Network topology:** A complete graph over generator nodes $\{1, \ldots, n\}$ and the substation $\{0\}$. Each edge $e \in \mathcal{E}$ has a continuous investment cost $\text{INV}(e)$ per MW of capacity built.

**Objective:**

$$\min \sum_{e \in \mathcal{E}} \text{INV}(e) \cdot \text{cap}(e) + G \cdot P$$

where $P = 0$ in the base formulation (generation cost suppressed), so the problem reduces to:

$$\min \sum_{e \in \mathcal{E}} \text{INV}(e) \cdot \text{cap}(e)$$

**Constraints:**

1. **Power flow balance** at each generator node $i \in S$ and at the substation, for each time period $t \in T$:

$$\sum_{l \in \delta^+(i)} f_{lt} - \sum_{l \in \delta^-(i)} f_{lt} = s_i \cdot g_{i,t} \quad \forall i \in S, \forall t \in T$$

2. **Line capacity limits:**

$$-F_l \leq f_{lt} \leq F_l \quad \forall l \in L, \forall t \in T$$

3. **Peak injection lower bound:**

$$G \geq \sum_{i \in S} s_i \cdot g_{i,t} \quad \forall t \in T$$

4. **Non-negativity:** $F_l \geq 0$, $f_{lt} \in \mathbb{R}$, $G \geq 0$

**Economic insight:** The optimal line capacities are exactly the peak flow requirements under the least-cost routing. Anti-correlated generators (e.g., one generates at night, one during the day) can share line capacity, dramatically reducing total investment.

---

### Module 4 — Cooperative Game Theory and Cost Allocation via the Nucleolus

This is the centerpiece applied topic of the course. The question is: when generators cooperate to share transmission infrastructure, how should total costs be fairly divided?

#### The Core

A cost allocation $\mathbf{x}$ is in the **core** if no coalition $S \subsetneq N$ has an incentive to break away:

$$\sum_{i \in S} x_i \leq C(S) \quad \forall S \subsetneq N, \qquad \sum_{i \in N} x_i = C(N)$$

The core can be empty or large. Among all core allocations, the **nucleolus** is the unique allocation that lexicographically maximizes the minimum excess across coalitions.

#### The Excess and the Nucleolus

The **excess** of coalition $S$ under allocation $\mathbf{x}$ measures how much $S$ saves relative to acting alone:

$$e(S, \mathbf{x}) = C(S) - \sum_{i \in S} x_i$$

The nucleolus solves a sequence of LPs:

**Step 1:** Maximize the minimum excess:

$$\max_{\mathbf{x}, \varepsilon} \; \varepsilon \quad \text{s.t.} \quad e(S, \mathbf{x}) \geq \varepsilon \;\; \forall S, \quad \sum_i x_i = C(N)$$

Let $\varepsilon^*$ be the optimal value. All coalitions achieving $e(S, \mathbf{x}) = \varepsilon^*$ are **frozen**.

**Step 2 (Lexicographic refinement):** Fix the frozen coalitions' excess at $\varepsilon^*$, then maximize the next smallest excess among remaining coalitions. Repeat until all generators are fixed.

**Julia implementation (sequential LP):**
```julia
function compute_nucleolus(C_dict, N_players)
    active_coalitions = [S for S in keys(C_dict) if 0 < length(S) < length(N_players)]
    fixed_excesses = Dict()
    while !isempty(active_coalitions)
        model = Model(HiGHS.Optimizer)
        @variable(model, x[N_players])
        @variable(model, e)
        @constraint(model, sum(x[i] for i in N_players) == C_dict[N_players])
        for S in active_coalitions
            @constraint(model, C_dict[S] - sum(x[i] for i in S) >= e)
        end
        for S in keys(fixed_excesses)
            @constraint(model, C_dict[S] - sum(x[i] for i in S) == fixed_excesses[S])
        end
        @objective(model, Max, e)
        optimize!(model)
        # identify newly binding coalitions and freeze them
        ...
    end
end
```

---

### Project 1 — Warm-Up: Perfect Anti-Correlation (n=2, T=2)

**Setup:**
- 2 generators, 2 time periods
- $g_1 = [0, 1]$, $g_2 = [1, 0]$ (perfectly anti-correlated)
- $\text{INV}(1,0) = \$90$, $\text{INV}(2,0) = \$100$, $\text{INV}(1,2) = \$50$

**Results:**

| Coalition | Cost ($/yr) |
|---|---|
| C({1}) | $90.00 |
| C({2}) | $100.00 |
| C({1,2}) | $120.00 |
| Standalone sum | $190.00 |
| **Total savings** | **$70.00** |

**Nucleolus allocation:**

| Generator | x* ($/yr) | Share of C(N) | Saves | Reduction |
|---|---|---|---|---|
| Gen 1 | $55.00 | 45.8% | $35.00 | 38.9% |
| Gen 2 | $65.00 | 54.2% | $35.00 | 35.0% |
| Sum | $120.00 | = C(N) | $70.00 | |

**Economic interpretation:** Perfect anti-correlation means the two generators never peak simultaneously. The shared line sees a maximum flow of only 1 MW (instead of 2 MW if they peaked together), allowing the coalition to build substantially less capacity. The nucleolus equalizes the absolute dollar savings ($35/yr each), reflecting the symmetric value each generator contributes to the partnership.

---

### Project 2 — Scaling: n=3 and n=10, T=n

Generators are placed at random coordinates; investment cost for each line is the Euclidean distance between nodes. Each generator peaks in exactly one time period ($g_i = e_i$, the $i$-th standard basis vector).

**n=3 results:**

| Generator | Allocation x* | Standalone | Saves | Saves % |
|---|---|---|---|---|
| Gen 1 | 1.47 | 3.16 | 1.70 | 53.6% |
| Gen 2 | 2.60 | 4.12 | 1.53 | 37.1% |
| Gen 3 | 3.06 | 5.83 | 2.78 | 47.5% |
| **Total** | **7.12** | **13.12** | **5.99** | **45.7%** |

**n=10 results summary:** Total savings jump to **79.1%** ($54.39 saved out of $68.69 standalone). Each generator individually saves between 63% and 86% of its standalone cost.

**Key finding:** Cooperative savings grow superlinearly with the number of generators. Larger coalitions unlock more network-sharing opportunities — distant generators can route power through intermediate nodes rather than building direct-to-hub lines.

---

### Project 3 — Correlation Study: n=2, T=168 Hours

**Setup:** Stochastic generation profiles simulated using a **Gaussian Copula** (Mersenne Twister PRNG, seed 42) to prescribe correlation $\rho \in [-1, 1]$ while maintaining Uniform[0, 100 MW] marginals for both generators. $T = 168$ time periods (one week of hourly data).

**Key results across $\rho$:**

| ρ | C({1,2}) ($) | Savings ($) | x₁*/C(N) | x₂*/C(N) |
|---|---|---|---|---|
| -1.0 | 5,000,000 | 4,857,474 | 0.5025 | 0.4975 |
| 0.0 | 9,485,605 | 389,155 | 0.5004 | 0.4996 |
| 1.0 | 9,882,829 | 0.00 | 0.5000 | 0.5000 |

**Key findings:**

1. **Monotonic decay of savings:** Cooperative savings decrease monotonically as $\rho \to 1$. Anti-correlated generators can share peak capacity; perfectly correlated generators peak simultaneously and gain nothing from cooperation.

2. **Stability of the nucleolus allocation:** Because both generators have identical marginal capacity distributions (Uniform[0, 100 MW]), the nucleolus allocates a nearly exact 50/50 cost share across *all* correlation levels. This is a theoretically important result — the nucleolus is determined by the symmetry of the marginals, not by the joint dependency structure.

3. **Peak-shaving efficiency:** At $\rho = -1$, the grand coalition needs only ~5 MW of line capacity (the two peaks never overlap); at $\rho = +1$, the required capacity approaches the sum of the individual capacities, eliminating all cooperative gains.

4. **Grid planning implication:** There is large financial value in resource diversity. Integrating anti-correlated renewables (e.g., solar + wind in complementary weather regions) reduces transmission expansion costs dramatically.

---

### Project 4 — Binary Investment Decisions (MILP)

**Motivation:** Real transmission infrastructure comes in discrete units — you either build a line or you don't. This project replaces the continuous capacity variable with a binary investment decision.

**Modified formulation:** Each edge $e \in \mathcal{E}$ has a fixed capacity $K^{\text{fixed}}$ and a binary investment variable $y_e \in \{0, 1\}$:

$$\min \sum_{e \in \mathcal{E}} y_e \cdot \text{INV}_e \cdot K^{\text{fixed}} + P \cdot G$$

Subject to:
$$f_{e,t} \leq y_e \cdot K^{\text{fixed}}, \quad f_{e,t} \geq -y_e \cdot K^{\text{fixed}}, \quad y_e \in \{0,1\}$$

**Consequences of binary variables:**
- The problem becomes a **MILP** — convexity is lost
- The solver must build a whole unit or nothing ("lumpiness"), leading to overbuilding relative to the continuous optimum
- All coalition costs are weakly higher under the binary model
- The core condition $\varepsilon^* > 0$ is still sufficient for cooperation to be stable
- Cost allocation shares shift slightly but cooperative incentives are preserved

**Comparative results (n=2, K=1.0, g₁₂=g₂₁=1):**

| | Continuous | Binary |
|---|---|---|
| Lines built | 3 × 0.5 MW | 2 × 1.0 MW |
| C({1,2}) | $120/yr | ~$130/yr |
| Cost shares | ≈50/50 | Slight shift |

**Scaling (n=10):** The binary model builds 10 large "hub-and-spoke" lines rather than the 11 smaller lines chosen by the continuous model. All coalition costs increase, but the nucleolus allocation remains qualitatively similar.

---

### Module 5 — Risk-Averse Renewable Portfolio Optimization

The later portion of the course extends into stochastic and risk-averse optimization for renewable energy procurement. This involves:

- **Multi-contract portfolio design:** Choosing a mix of renewable energy contracts (wind, solar, etc.) to minimize expected cost while controlling downside risk
- **CVaR (Conditional Value at Risk):** A coherent risk measure used to penalize worst-case outcomes beyond the Value at Risk threshold
- **Risk-averse stochastic programming:** Minimizing a weighted combination of expected cost and CVaR over a set of scenarios
- **Excel-based decision tools:** Scenario generation and portfolio analysis implemented in spreadsheet models for practical use

The key tradeoff explored is between **cost efficiency** (minimize expected procurement cost) and **risk aversion** (limit exposure to high-cost scenarios under renewable generation shortfalls).

---

## Problem Set Structure

Projects build progressively in complexity:

| Project | Topic | Methods | Scale |
|---|---|---|---|
| Project 1 | Transmission cost allocation, nucleolus derivation | LP, cooperative game theory | n=2, T=2 |
| Project 2 | Scaling the nucleolus algorithm | LP, sequential nucleolus | n=3, n=10 |
| Project 3 | Stochastic correlated generation | LP + copula simulation | n=2, T=168 |
| Project 4 | Binary investment, MILP formulation | MILP, nucleolus under non-convexity | n=2, n=3, n=10 |

---

## Key Theoretical Results Developed in the Course

**1. Convexity of the pointwise maximum:**
The max of finitely many convex functions is convex. Used throughout to verify that LP objective functions and cutting-plane approximations remain in the convex class.

**2. Kelley convergence:**
The cutting-plane lower bound satisfies $f_k \leq f_{k+1} \leq f$, ensuring monotone improvement. Convergence to $\varepsilon$-optimality is guaranteed for polyhedral $X$ and convex $f$.

**3. Nucleolus existence and uniqueness:**
The nucleolus always exists and is unique for any characteristic function game with a non-empty core. It is always in the core (when the core is non-empty).

**4. Symmetry of the nucleolus:**
If all generators have symmetric marginal distributions, the nucleolus splits costs equally regardless of the joint correlation structure. This makes it a robust and envy-free allocation rule.

**5. MILP and cooperative stability:**
Even when the investment problem is non-convex (binary decisions), cooperation remains stable if and only if $\varepsilon^* > 0$ at the nucleolus — i.e., every coalition still saves by participating in the grand coalition.

---

## Tools and Libraries Reference

```julia
# Standard project imports
import Pkg
Pkg.add("JuMP")
Pkg.add("HiGHS")
Pkg.add("Plots")
Pkg.add("StatsPlots")

using JuMP, HiGHS, Plots, Printf, StatsPlots

# Define and solve an LP
model = Model(HiGHS.Optimizer)
set_silent(model)
@variable(model, x >= 0)
@objective(model, Min, 3x)
@constraint(model, x >= 1)
optimize!(model)
println(value(x))  # => 1.0
```

---

## Academic Context

ECON 238 sits at the intersection of three fields:

- **Operations Research / Mathematical Programming** — LP, MILP, cutting-plane methods, subgradient theory
- **Energy Economics** — Transmission network design, renewable integration policy, capacity markets
- **Cooperative Game Theory** — Cost allocation, the core, the nucleolus, stability analysis

The applications are drawn from real challenges in power system decarbonization: how to fairly allocate shared transmission costs among wind and solar developers, how binary infrastructure investments change cooperative incentives, and how correlated generation profiles affect the value of grid diversity. These are active research and policy questions relevant to FERC interconnection proceedings, state renewable portfolio standards, and ISO/RTO cost allocation tariffs.

---

*Course materials, projects, and code developed as part of ECON 238, Stanford University.*
