using Bridge
using Plots

struct OU <: Bridge.ContinuousTimeProcess{Float64}
    μ::Float64
end
Bridge.b(s, x, P::OU) = -P.μ*x
Bridge.σ(s, x, P::OU) = 1.0


function Λ(x)
    return max(0.0, 1.0 - 2abs(x - 0.5))
end

function Λ(t, l, k)
    x = mod(t, 1)
    Λ((x)*2^(l-1) -k + 1)
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

function MH(t, iterations, ρ, L, P, Q)
    count = 0
    Z = BBridgeCies(t, L)
    lgir = -Inf
    for i in 1:iterations
        Z1 = BBridgeCies(t, L)
        z° = sqrt(ρ)*Z.yy + sqrt(1-ρ)*Z1.yy #preconditioned Crank-Nicolson
        Z° = SamplePath(t, z°)
        lgir° = Bridge.girsanov(Z°, P, Q)
        A = exp(lgir° - lgir)
        if rand() < A
            Z = Z°
            lgir = lgir°
            count += 1
        end
    end
    Z, count
end

t = 0:.001:1
L = 10
iterations = 100
ρ = .7 #number 0<=ρ<=1
Q = Wiener()
P = OU(10.0)

iter1 = 1
B, iter = MH(t, iter1, ρ, L, P, Q)
plot(B)

iter2 = 10
B, iter = MH(t, iter2, ρ, L, P, Q)
plot(B)

iter3 = 1000
B, iter = MH(t, iter3, ρ, L, P, Q)
plot(B)
