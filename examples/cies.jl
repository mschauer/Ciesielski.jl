function Λ(x)
    return max(0.0, 1.0 - 2abs(x-0.5))
end

function Λ(t, l, k)
    x = mod(t, 1)
    Λ((x)*2^(l-1) -k+1 )
end
L = 10
Z = randn(2^(L)-1)

function cies(t, Z, L)
    x = 0.0
    i = 1
    for l in 1:L
        for k in 1:2^(l-1)
            x += 2^(-l/2-0.5)*Z[i]*Λ(t, l, k)
            i += 1
        end
    end
    x
end
 w = [cies(ti, Z, L) for ti in t]
 W = SamplePath(t, w)
 plot(W)
