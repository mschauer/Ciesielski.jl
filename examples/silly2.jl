
using Bridge
using Plots
using Colors
t= 0:.001:1
struct OU <: Bridge.ContinuousTimeProcess{Float64}
    μ::Float64
end
Bridge.b(s, x, P::OU) = -P.μ*x + 1.0
Bridge.σ(s, x, ::OU) = 1.0
P = OU(1.0)
plot(t, NaN*t, legend=false)
for i in 1:20
    W = sample(t, Wiener())
    ll = Bridge.girsanov(W, P, Wiener())
    c = min(1.0, 0.5*exp(ll)) #number between zeo and one
    display(plot!(W, color= RGB(c, 0.5, 1-c), linewidth=0.3))
end
