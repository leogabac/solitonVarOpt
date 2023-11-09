println("======================================================================")
println("Loading packages...")

using LinearAlgebra, Distributed # Basic libraries
using FiniteDiff # Differentiation, gradients, hessian
using DelimitedFiles # Plotting, and files
using ProgressMeter # Other cool features
addprocs(4)
@everywhere using ForwardDiff
@everywhere using FiscomTools

# =============================================================================
# Auziliary functions
# =============================================================================
@everywhere lambda=1.0;

println("λ = $lambda")

println("Defining functions...")
@everywhere norm2(x::AbstractVector)::Real = sum( abs2.(x) );
@everywhere abs4(x) = abs(x)^4;


# =============================================================================
# Beams
# =============================================================================

"""
    LaguerreGaussBeam(x, y, z, w0, phi, lambda, l, p)
    Laguerre-Gaussian beam
"""
function LaguerreGaussBeam(x::Real, y::Real, z::Real, w0::Real, phi::Real, lambda::Real, l, p)
    LG::ComplexF64=0.0 + im*0.0

    zr = pi*(w0^2)/lambda
    wz2 = (w0^2) * (1 + (z/zr)^2)
    wz = sqrt(wz2)
    rr2=(x^2 + y^2)/(wz2)
    C = sqrt(2 * factorial(p) / (pi*factorial(p + abs(l))))
    k = 2*pi/lambda

    if p==0
        LG = (C/wz) * ((sqrt(2*rr2))^abs(l)) * exp(-rr2) *
            1.0 * exp(-im*rr2*z/zr) * exp(im*(l*atan(y,x)+phi)) *
            exp(-im*(2*p+abs(l)+1) * atan(z,zr))
    else
        LG = (C/sqrt(wz)) * ((sqrt(2*rr2))^abs(l)) * exp(-rr2) *
            LaguerrePoly.(p, abs(l), 2*rr2) * exp(-im*rr2*z/zr) * exp(im*(l*atan(y,x)+phi)) *
            exp(-im*(2*p+abs(l)+1) * atan(z,zr))
    end

    return LG::ComplexF64
end

"""
    HermiteGaussBeam(x, y, z, w0, phi, lambda, m, n)
    Hermite-Gaussian beam
"""
function HermiteGaussBeam(x::Real, y::Real, z::Real, w0::Real, phi::Real, lambda::Real, m::Integer, n::Integer)
    HG::ComplexF64=0.0 + im*0.0

    zr = pi*(w0^2)/lambda
    wz2 = (w0^2) * (1 + (z/zr)^2)
    wz = sqrt(wz2)
    rr2=(x^2 + y^2)/(wz2)
    C = sqrt(2/pi) * (2.0)^(-(m+n)/2) * (1/sqrt(factorial(n)*factorial(m)*(w0^2)))
    k = 2*pi/lambda

    HG= C * HermitePoly(m,sqrt(2)*x/wz) * HermitePoly(n,sqrt(2)*y/wz) * exp(-im*rr2*z/zr) *
        exp(-rr2) * exp(-im*(m+n+1)*atan(z,zr)) * exp(im*(phi))   # Im not sure about adding the dot in LaguerrePoly...
    return HG::Complex{Float64}
end

# =============================================================================
# Gradients, Hessian, etc...
# =============================================================================

# Spatial gradients
@everywhere grad(U,x::Real,A::AbstractVector)::Vector = ForwardDiff.gradient( r-> U(r[1], A),[x] );
@everywhere grad(U,x::Real,y::Real,A::AbstractVector)::Vector = ForwardDiff.gradient( r-> U(r[1],r[2], A),[x,y] );
@everywhere grad(U,x::Real,y::Real,z::Real,A::AbstractVector)::Vector = ForwardDiff.gradient( r-> U(r[1],r[2],r[3], A),[x,y,z] );

# Parameter gradient
@everywhere grad(S,A::AbstractVector)::Vector = FiniteDiff.finite_difference_gradient(S,A);


# Hessian
H(S,A::AbstractVector)::Matrix = FiniteDiff.finite_difference_hessian(S,A);


# =============================================================================
# Integration techniques
# =============================================================================


@everywhere function blockEvaluation(f,rx,dx,A::AbstractVector)
    sum(f(xi,A) for xi in rx)*dx
end

@everywhere function blockEvaluation(f,rx,ry,dx,A::AbstractVector)
    sum(f(xi,yi,A) for yi in ry, xi in rx)*dx^2
end

@everywhere function blockEvaluation(f,rx,ry,rz,dx,A::AbstractVector)
    sum(f(xi,yi,zi,A) for zi in rz, yi in ry, xi in rx)*dx^3
end

function DistributedRiemannSum(f,A::AbstractVector,n::Int64)
    N = 2^9
    mesh = range(-100,stop = 100,length=N)
    Δx = step(mesh);
    mesh1,mesh2 = Iterators.partition(mesh,div(N,2)) |> Tuple

    if n == 1
        f1 = @spawnat :any blockEvaluation(f,mesh1,Δx,A);
        f2 = @spawnat :any blockEvaluation(f,mesh2,Δx,A);
        return fetch(f1) + fetch(f2) 
    elseif n==2
        f1 = @spawnat :any blockEvaluation(f,mesh1,mesh1,Δx,A);
        f2 = @spawnat :any blockEvaluation(f,mesh1,mesh2,Δx,A);
        f3 = @spawnat :any blockEvaluation(f,mesh2,mesh1,Δx,A);
        f4 = @spawnat :any blockEvaluation(f,mesh2,mesh2,Δx,A);
        return fetch(f1) + fetch(f2) + fetch(f3) + fetch(f4)
    else
        f1 = @spawnat :any blockEvaluation(f,mesh1,mesh1,mesh1,Δx,A);
        f2 = @spawnat :any blockEvaluation(f,mesh1,mesh2,mesh1,Δx,A);
        f3 = @spawnat :any blockEvaluation(f,mesh2,mesh1,mesh1,Δx,A);
        f4 = @spawnat :any blockEvaluation(f,mesh2,mesh2,mesh1,Δx,A);
        f5 = @spawnat :any blockEvaluation(f,mesh1,mesh1,mesh2,Δx,A);
        f6 = @spawnat :any blockEvaluation(f,mesh1,mesh2,mesh2,Δx,A);
        f7 = @spawnat :any blockEvaluation(f,mesh2,mesh1,mesh2,Δx,A);
        f8 = @spawnat :any blockEvaluation(f,mesh2,mesh2,mesh2,Δx,A);
        return fetch(f1) + fetch(f2) + fetch(f3) + fetch(f4) + fetch(f5) + fetch(f6) + fetch(f7) + fetch(f8)
    end
end

# =============================================================================
# Lagrangian density, and lagrangians
# =============================================================================


@everywhere s = 0.5;
# Non linear medium
@everywhere NL(U,A,x) = log(s*abs2(U(x,A)) + 1)/s^2 -  abs2(U(x,A))/s;
@everywhere NL(U,A,x,y) =  log(s*abs2(U(x,y,A)) + 1)/s^2 -  abs2(U(x,y,A))/s;
@everywhere NL(U,A,x,y,z) = log(s*abs2(U(x,y,z,A)) + 1)/s^2 -  abs2(U(x,y,z,A))/s;

"""
    Computes the lagrangian density of the GNLSE.
"""

@everywhere LagrangianDensity(x::Real,A::AbstractVector) = lambda*abs2(U(x,A)) + norm2(grad(realU,x,A) + im*grad(imagU,x,A)) + NL(U,A,x); 

@everywhere LagrangianDensity(x::Real,y::Real,A::AbstractVector) = lambda*abs2(U(x,y,A)) + norm2(grad(realU,x,y,A) + im*grad(imagU,x,y,A)) + NL(U,A,x,y);

@everywhere LagrangianDensity(x::Real,y::Real,z::Real,A::AbstractVector) = lambda*abs2(U(x,y,z,A)) + norm2(grad(realU,x,y,z,A) + im*grad(imagU,x,y,z,A)) + NL(U,A,x,y,z);

"""
    Computes the Lagrangian. Integrates over space the lagrangian density. 1D in real space
"""
function Lagrangian1(A::AbstractVector)
    return DistributedRiemannSum(LagrangianDensity,A,1)
end;

"""
    Computes the Lagrangian. Integrates over space the lagrangian density. 2D in real space
"""
function Lagrangian2(A::AbstractVector)
    return DistributedRiemannSum(LagrangianDensity,A,2)
end;

"""
    Computes the Lagrangian. Integrates over space the lagrangian density. 3D in real space
"""
function Lagrangian3(A::AbstractVector)
    return DistributedRiemannSum(LagrangianDensity,A,3)
end;


# Ansatz and some other parts
@everywhere U(x::Real,A::AbstractVector) = A[1] * exp( -(x^2)/(A[2]^2) );

@everywhere m=0;
#@everywhere U(x::Real,y::Real,A::AbstractVector) = A[1]*exp( -(x^2+y^2)/(A[2]^2) ) * cis(m*atan(y,x)) * sqrt((x^2+y^2))^m;
@everywhere U(x::Real,y::Real,A::AbstractVector) = A[1]*exp( -(x^2+y^2)/(A[2]^2) );

@everywhere U(x::Real,y::Real,z::Real,A::AbstractVector) = A[1] * exp( -(x^2+y^2+z^2)/(A[2]^2) );

@everywhere realU(x::Real,A::AbstractVector) = real(U(x,A)) ;
@everywhere imagU(x::Real,A::AbstractVector) = imag(U(x,A)) ;

@everywhere realU(x::Real,y::Real,A::AbstractVector) = real(U(x,y,A)) ;
@everywhere imagU(x::Real,y::Real,A::AbstractVector) = imag(U(x,y,A)) ;

@everywhere realU(x::Real,y::Real,z::Real,A::AbstractVector) = real(U(x,y,z,A)) ;
@everywhere imagU(x::Real,y::Real,z::Real,A::AbstractVector) = imag(U(x,y,z,A)) ;


# =============================================================================
# Newton-Rhapsonvt5
# =============================================================================

"""
    Newton-Raphson method to find stationary points.
"""
function optimize(f,A::AbstractVector ;iter=30, TOL = 1e-5)
   
    println("Initial grad= $(norm(grad(f,A)))")

    @showprogress for k in 1:iter

        #println("- $k out of $iter -" )
        
        A = A - inv(H(f,A)) * grad(f,A)
        
        # Penalization to parameters close to zero
        close2zero = map( c -> isapprox(c,0.0,atol=1e-2), A)
        A = A + close2zero .* rand(-10:0.1:10,length(A))
        
        #println("A = $A")
        #println("grad= $(norm(grad(f,A)))")
    end
    println("Final grad= $(norm(grad(f,A)))")
    return A
end;


# =============================================================================
# Experiments
# =============================================================================

A0 = 1 .+ ones(2);

lloracion = 10
@time result = optimize(Lagrangian2,A0, iter = lloracion);
result = abs.(result);

println("A = $result")

println("Writing parameters...")
open("m0_params.csv","a") do fh
    towrite = hcat([lambda s],result')
    writedlm(fh,towrite,',');
end

