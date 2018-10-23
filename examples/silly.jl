
using Bridge
using Plots
t= 0:.001:1
W = sample(t, Wiener())
plot(W.tt, W.yy)
struct OU <: Bridge.ContinuousTimeProcess{Float64}
    μ::Float64
end
Bridge.b(s, x, P::OU) = -P.μ*x
Bridge.σ(s, x, ::OU) = 1.0
P = OU(1.0)
Bridge.girsanov(W, P, Wiener())
