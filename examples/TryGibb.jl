using Bridge
using Plots
import LinearAlgebra.dot
using Distributions
#import Bridge.sample

struct OU <: Bridge.ContinuousTimeProcess{Float64}
    μ::Float64
    α::Float64
end
Bridge.b(s, x, P::OU) = P.α*(P.μ - x)
Bridge.σ(s, x, P::OU) = 1.0
function Bridge.sample(tt,P::OU)
    dt = tt[2:end]  .-  tt[1:end-1]
    x = zeros(1,length(tt))
    for i = 1:length(t)-1
        x[i+1] = x[i]+P.α*(P.μ-x[i])*dt[i]+sqrt(dt[i])*randn();
    end
    x=vec(x)
    X = Bridge.SamplePath(tt,x)
    return X
end

#suppose for the moment that t_0= 0 t_1 = 1
struct GBBCies <: Bridge.ContinuousTimeProcess{Float64}
    u::Float64
    v::Float64
    L::Int16
end
function Bridge.sample(tt,P::GBBCies)
    L=P.L
    u=P.u
    v=P.v
    W = BBridgeCies(tt, L).yy .+ (v - u)/(tt[end] - tt[1]).*tt .+ u
    W = SamplePath(tt,W)
end


Bridge.b(s, x, P::GBBCies) = (P.v - P.u)
Bridge.σ(s, x, P::GBBCies) = 1.0

function interpolate(x1, y1, x2, y2, x)
    y = y1 .+ (y2 .- y1)/(x2 .- x1).*(x .- x1)
end



function Λ(x)
    return max(0.0, 1.0 - 2abs(x - 0.5))
end
function Λ(t, l, k)
    x = mod(t, 1)
    Λ((x)*2^(l-1) - k + 1)
end
function cies(t, Z, L)
    x = 0.0
    i = 1
    for l in 1:L
        for k in 1:2^(l-1)
            x += 2^(-l/2 - 0.5)*Z[i]*Λ(t, l, k)
            i += 1
        end
    end
    x
end
function BBridgeCies(tt, L)
    Z = randn(2^(L) - 1)
    Z = [cies(ti, Z, L) for ti in tt]
    Z = SamplePath(tt, Z)
end
function Bridge.sample(tt,P::GBBCies)
    L=P.L
    u=P.u
    v=P.v
    W = BBridgeCies(tt, L).yy .+ (v - u)/(tt[end] - tt[1]).*tt .+ u
    W = SamplePath(tt,W)
end

#############################################
################ APPLICATION ################
#############################################
dx=0.01
t = 0:dx:2
μtrue = 10.0
X = Bridge.sample(t,OU(μtrue,1.0))
plot(X) #check
index = [1,101,201]
X_obs = SamplePath(X.tt[index], X.yy[index])
plot(X_obs;seriestype=:scatter) #check
#initialize Z
Z=SamplePath(t,zeros(length(t)))
#linear interpolation
for j in 1:length(index)-1
            Z.yy[index[j]:index[j+1]] = interpolate(X_obs.tt[j],X_obs.yy[j],X_obs.tt[j+1],X_obs.yy[j+1],t[index[j]:index[j+1]])
end
plot(Z) #check
μstart= 5
function GS(index, X_obs, μstart, t, iterations, ρ, σ0,μ0)
    μ = μstart
    A = [μ]
    #inizialize Z with linear interpolation
    Z=SamplePath(t,zeros(length(t)))
    for j in 1:length(index)-1
                Z.yy[index[j]:index[j+1]] = interpolate(X_obs.tt[j],X_obs.yy[j],X_obs.tt[j+1],X_obs.yy[j+1],t[index[j]:index[j+1]])
    end
    lgir = repeat([-Inf], length(index)-1)
    for i in 1:iterations
        for j in length(index)-1
            # first segment
            u = X_obs.yy[j]
            v = X_obs.yy[j+1]
            ii = index[j]:index[j+1]
            tt = t[ii]
            Z1 = sample(tt, GBBCies(u,v,10)) #updating path
            z° = sqrt(ρ)*Z.yy[ii] + sqrt(1-ρ)*Z1.yy #preconditioned Crank-Nicolson
            Z° = SamplePath(tt, z°)
            lgir° = Bridge.girsanov(Z°, OU(μ,1.0), Wiener())
            A1 = exp(lgir° - lgir[j])
                if rand() < A1
                    Z.yy[ii] = Z°.yy
                    lgir[j] = lgir°
                end
            end

        #Step 2
        T=t[end]
        σpost= 1 / (T + (1\σ0))
        Δt = t[2:end].-t[1:end-1]
        integ = dot(Z.yy[1:end-1],Δt) # dot operator dt and Z[1:end-1]
        μpost= (X_obs.yy[end] + integ + μ0/σ0)*σpost
        μ = rand(Normal(μpost,σpost)) #updating parameter
        push!(A,μ)
    end
    A
end


GS(index, X_obs, 5.0,t, 10, 0.4,.1,4.0)
