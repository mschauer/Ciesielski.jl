using Bridge
using Plots
import LinearAlgebra.dot
using Distributions


struct OU <: Bridge.ContinuousTimeProcess{Float64}
    μ::Float64
    α::Float64
end
Bridge.b(s, x, P::OU) = P.α*(P.μ - x)
Bridge.σ(s, x, P::OU) = 1.0

struct GBBCies <: Bridge.ContinuousTimeProcess{Float64}
    u::Float64
    v::Float64
    L::Int16
end
function Bridge.sample(tt,P::GBBCies)
    L = P.L
    u = P.u
    v = P.v
    W = BBridgeCies(tt, L).yy #.+ (v - u)/(tt[end] - tt[1]).*(tt .- tt[1])
    W = SamplePath(tt,W)
end

function interpolate(x1, y1, x2, y2, x)
    return ( y1 .+ (y2 .- y1)/(x2 .- x1).*(x .- x1))
end

function Λ(x,t)
    return max(0.0, t - 2abs(x - t/2))
end

function Λ(t, l, k, T)
    x = mod(t, T)   #ok
    Λ((x)*(2)^(l-1) -T*(k - 1),T)
end

function cies(t, Z, L, T)
    x = 0.0
    i = 1
    for l in 1:L
        for k in 1:2^(l-1)
            x += 2^(-l/2 - 0.5)*Z[i]*Λ(t, l, k, T)
            i += 1
        end
    end
    x
end

function BBridgeCies(tt, L)
    t = tt .- tt[1]
    T = t[end]
    Z = randn(2^(L) - 1)
    Z = [cies(ti, Z, L, T) for ti in t]
    Z = SamplePath(tt, Z)
end


function Bridge.sample(tt,P::GBBCies)
    L = P.L
    u = P.u
    v = P.v
    W = BBridgeCies(tt, L).yy .+ (v - u)/(tt[end] - tt[1]).*tt .+ u
    W = SamplePath(tt,W)
end

#############################################
################ APPLICATION ################
#############################################
dx = 0.001
t = 0:dx:1
μtrue = 4.0
X = solve(EulerMaruyama(), 0.0, sample(t, Wiener()), OU(μtrue, 1.0))
plot(X) #check
index = [1, 400, 600, 1001]
X_obs = SamplePath(X.tt[index], X.yy[index])
plot(X_obs; seriestype=:scatter) #check

μstart= 5.0
function GS(index, X_obs, μstart, t, iterations, ρ, σ0,μ0)
    t = t[1:index[end]]
    μ = μstart
    A = [μ]
    #inizialize Z with linear interpolation
    Z = SamplePath(t,zeros(length(t)))
    for j in 1:length(index) - 1
                Z.yy[index[j]:index[j + 1]] = interpolate(X_obs.tt[j], X_obs.yy[j], X_obs.tt[j + 1], X_obs.yy[j + 1], t[index[j]:index[j + 1]])
    end
    lgir = fill(-Inf, length(index) - 1)
    for i in 1:iterations
        for j in length(index) - 1
            # first segment
            u = X_obs.yy[j]
            v = X_obs.yy[j+1]
            ii = index[j]:index[j+1]
            tt = t[ii]
            Z1 = sample(tt, GBBCies(u, v, 10)) #updating path
            z° = sqrt(ρ)*Z.yy[ii] + sqrt(1 - ρ)*Z1.yy #preconditioned Crank-Nicolson
            Z° = SamplePath(tt, z°)
            lgir° = Bridge.girsanov(Z°, OU(μ,1.0), Wiener())
            A1 = exp(lgir° - lgir[j])
                if rand() < A1
                    Z.yy[ii] = Z°.yy
                    lgir[j] = lgir°
                end
            end

        #Step 2
        T = t[end]
        σpost= 1 / (T + (1\σ0))
        Δt = diff(t)
        μpost = (X_obs.yy[end] + dot(Z.yy[1:end-1], Δt) + μ0/σ0)*σpost
        μ = rand(Normal(μpost, σpost)) #updating parameter
        push!(A,μ)
    end
    A
end

μstart = 1.0
iterations = 1000
A = GS(index, X_obs, μstart ,t , iterations, 0.7, 10.0, μstart)
plot(1:iterations + 1, A)
hline!([μtrue])
