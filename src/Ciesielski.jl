module Ciesielski
    using Bridge
    using Distributions
    export StandardizeBridge, GeneralizeBridge, interpolate, BBCies

    function Λ(x,T)
        return max(0.0, T - 2abs(x - T/2))
    end
    function Λ(x, l, k, T)
        x = mod(x, T)   #ok
        Λ((x)*(2)^(l-1) -T*(k - 1),T)
    end
    function ϕ1(x,u,T)
        return u*(T-x)/T
    end
    function ϕ2(x,v,T)
        return v*x/T
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
    function BBCies(tt, L, u, v)
        t = tt .- tt[1]
        T = t[end]
        Z = randn(2^(L) - 1)
        Z = [cies(ti, Z, L, T) + ϕ1(ti, u, T) + ϕ2(ti, v, T) for ti in t]
        Z = SamplePath(tt, Z)
    end
    function StandardizeBridge(P::SamplePath)
        tt = P.tt .- P.tt[1]
        T = tt[end]
        v = P.yy[end]
        u = P.yy[1]
        W = P.yy - [ϕ1(x, u, T) + ϕ2(x, v, T) for x in tt]
        W = SamplePath(P.tt, W)
    end
    function GeneralizeBridge(P::SamplePath,u,v)
        tt = P.tt .- P.tt[1]
        T = tt[end]
        W = P.yy + [ϕ1(x, u, T) + ϕ2(x, v, T) for x in tt]
        W = SamplePath(P.tt, W)
    end
    function interpolate(x1, y1, x2, y2, x)
        return ( y1 .+ (y2 .- y1)/(x2 .- x1).*(x .- x1))
    end
end  # module Ciesielski
