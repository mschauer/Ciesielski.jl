using Ciesielski
using Plots
using Bridge
using Distributions
import LinearAlgebra.dot

struct OU <: Bridge.ContinuousTimeProcess{Float64}
    μ::Float64
    α::Float64
end
Bridge.b(s, x, P::OU) = P.α*(P.μ - x)
Bridge.σ(s, x, P::OU) = 1.0



function GS(index, X_obs, μstart, t, iterations, ρ, σ0,μ0)
    t = t[1:index[end]]
    μ = μstart
    A = [μ]
    T = t[end]
    σpost= 1 / (T + (1\σ0)) #not depending on the path
    Δt = diff(t)
    #inizialize Z with linear interpolation
    Z = SamplePath(t,zeros(length(t)))
    for j in 1:length(index) - 1
                Z.yy[index[j]:index[j + 1]] = interpolate(X_obs.tt[j], X_obs.yy[j], X_obs.tt[j + 1], X_obs.yy[j + 1], t[index[j]:index[j + 1]])
    end
    lgir = fill(-Inf, length(index) - 1)
    for i in 1:iterations
        for j in length(index) - 1
            u = X_obs.yy[j]
            v = X_obs.yy[j+1]
            ii = index[j]:index[j+1]
            tt = t[ii]
            Z1 = BBCies(tt, 10, 0, 0)
            Zseg = SamplePath(tt,Z.yy[ii]) #L = 10
            z° = ρ*StandardizeBridge(Zseg).yy + sqrt(1 - ρ^2)*Z1.yy #preconditioned Crank-Nicolson
            Z° = SamplePath(tt, z°)
            Z° = GeneralizeBridge(Z°,u,v)

            lgir° = Bridge.girsanov(Z°, OU(μ,1.0), Wiener())
            A1 = exp(lgir° - lgir[j])
                if rand() < A1
                    Z.yy[ii] = Z°.yy
                    lgir[j] = lgir°
                end

            end

        #Step 2
        μpost = (X_obs.yy[end] + dot(Z.yy[1:end-1], Δt) + μ0/σ0)*σpost
        μ = rand(Normal(μpost, σpost)) #updating parameter
        push!(A,μ)
    end
    return (A)
end

dx = 0.01
t = 0:dx:100
μtrue = 4.0
X = solve(EulerMaruyama(), 0.0, sample(t, Wiener()), OU(μtrue, 1.0))
plot(X) #check
index = 1:10:10001
X_obs = SamplePath(X.tt[index], X.yy[index])
plot(X_obs; seriestype=:scatter) #check
μstart = 3.0
iterations = 50
println("The true value is μ = ", μtrue)
println("The number of iterations is:  ", iterations)
println("The starting value is μstart = ", μstart)
σ0 = 1.0 #algorithm is σ0 dependent need to be fixed
A = GS(index, X_obs, μstart ,t , iterations, 0.5, σ0 , 0.0)
plot(1:iterations + 1, A)
hline!([μtrue])
