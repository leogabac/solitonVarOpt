# =============================================================================
# Understanding the error in 1D
# =============================================================================

println("======================================================================")
println("Loading packages...")

using LinearAlgebra, Distributed # Basic libraries
using FiniteDiff # Differentiation, gradients, hessian
using DelimitedFiles # Plotting, and files
using ProgressMeter # Other cool features
addprocs(4)
@everywhere using ForwardDiff
@everywhere using FiscomTools
@everywhere using PolynomialInterpolations

# =============================================================================
# Auziliary functions
# =============================================================================
@everywhere lambda=0.5;

println("lambda = $lambda")

println("Defining functions...")
@everywhere norm2(x::AbstractVector)::Real = sum( abs2.(x) );
@everywhere abs4(x) = abs(x)^4;

# =============================================================================
# Polynomials
# =============================================================================


@everywhere function LaguerrePoly(n::Integer,alpha::Integer,x)
    sum( binomial(n+alpha,n-k) * (-x)^k / factorial(k) for k in 0:n)
end

@everywhere function HermitePoly(n::Integer,x)
    upper = floor(n/2) |> Int
    Hnt = sum( (-1)^k/factorial(k)/factorial(n-2k)*(2*x)^(n-2k) for k in 0:upper )
    return factorial(n) * Hnt
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

# Spatial second derivative
secondDer(U,x::Real,A::AbstractVector) = ForwardDiff.hessian( r-> U(r[1], A),[x]);




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
    N = 2^10
    mesh = range(-200,stop = 200,length=N)
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

# Non linear medium
@everywhere NL(U,A,x) = abs4(U(x,A))/2;
@everywhere NL(U,A,x,y) = abs4(U(x,y,A))/2;
@everywhere NL(U,A,x,y,z) = abs4(U(x,y,z,A))/2;

"""
    Computes the lagrangian density of the GNLSE.
"""

# Agregue un factor de 1/2 en 1D sobre el termino de la segunda derivada
@everywhere LagrangianDensity(x::Real,A::AbstractVector) = lambda*abs2(U(x,A)) + norm2(grad(realU,x,A) + im*grad(imagU,x,A))/2 - NL(U,A,x); 

@everywhere LagrangianDensity(x::Real,y::Real,A::AbstractVector) = lambda*abs2(U(x,y,A)) + norm2(grad(realU,x,y,A) + im*grad(imagU,x,y,A)) - NL(U,A,x,y);

@everywhere LagrangianDensity(x::Real,y::Real,z::Real,A::AbstractVector) = lambda*abs2(U(x,y,z,A)) + norm2(grad(realU,x,y,z,A) + im*grad(imagU,x,y,z,A)) - NL(U,A,x,y,z);

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
# @everywhere U(x::Real,A::AbstractVector) = A[1] * exp( -(x^2)/(A[2]^2) );
@everywhere U(x::Real,A::AbstractVector) = A[1] * exp( -abs(x) );
# @everywhere U(x::Real,A::AbstractVector) = A[1] * HermitePoly(1,x) * exp( -(x^2)/(A[2]^2) );
# @everywhere U(x::Real,A::AbstractVector) = A[1] * sech( x/A[2] );
# @everywhere U(x::Real,A::AbstractVector) = A[1] / (x^2 + A[2]^2);
# @everywhere U(x::Real,A::AbstractVector) = A[1] * exp( -(x^4)/(A[2]^4) );



@everywhere realU(x::Real,A::AbstractVector) = real(U(x,A)) ;
@everywhere imagU(x::Real,A::AbstractVector) = imag(U(x,A)) ;

# =============================================================================
# Newton-Rhapson
# =============================================================================


function is_any_zero(A::Vector)
    for (idx,element) in enumerate(A)
        if isapprox(element,0.0,atol=1e-2)
            return idx,true
        end
    end

    return 0,false
end

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
# Optimization
# =============================================================================

#A0 = rand(-10:0.01:10,2);
A0 = ones(1) .+ 10;
println("A = $A0")

lloracion = 50;
println("Doing optimization...")
@time result = optimize(Lagrangian1,A0, iter = lloracion);
result = abs.(result)

finalgrad = grad(Lagrangian1,result) |> norm

# =============================================================================
# Computing error
# =============================================================================


# -lambda*U(x,A) + segunda derivada de U /2 + abs2(U(x,A)) * U(x,A)
# -λU + ∂²xU / 2 + |U|*U
NLSE(U,A,x) = -lambda*U(x,A) + secondDer(U,x,A)[1] / 2 + abs2(U(x,A))*U(x,A)
N = 2^9;
mesh = range(-150,stop = 150,length=N);

function getepsilon(U,A,mesh)
    sum(NLSE(U,A,xi) for xi in mesh) / N
end

function average(x)
    sum(x)/length(x)
end


epsilon = getepsilon(U,result,mesh)
errors = [ getepsilon(U,rand(0.1:0.01:10,2),mesh) for _ in 1:1000 ]
average(errors)

# =============================================================================
# Saving stuff
# =============================================================================


println("A = $result")

mainPath = "C:/Users/leoga/Dropbox/Academic/MNT/projects/soliton variational/"
subpath = "data/1dparams_error/params_gaussian.csv"

path = joinpath(mainPath,subpath)

println("Writing parameters...")
open(path,"a") do fh
    towrite = hcat([lambda],result',[finalgrad epsilon])
    writedlm(fh,towrite,',');
end
