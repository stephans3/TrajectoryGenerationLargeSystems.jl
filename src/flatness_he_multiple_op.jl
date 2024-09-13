

using LinearAlgebra
using OrdinaryDiffEq

α_diff = 3; # Scale matrix A
β_in = 1;   # Scale vector B
N = 11; # Number of states

D = diagm(vcat(-1,-2ones(Int64, N-2),-1))
for n=1:N-1
    D[n,n+1] = 1;
    D[n+1,n] = 1;
end

# State space formulation
A = α_diff * D 
B = β_in * vcat(1, zeros(Int64,N-1))
C = vcat(zeros(Int64,N-1),1)'

# Transformation matrix
T = C
for n=1:N-1
    ca = C*A^n
    T = vcat(T, ca)
    println("C A^",n, " = ", ca)
end

for n=0:N-1
    cab = C*(A^n)*B
    println("C A^",n, " B = ", cab)
end


# Differentiate tanh
T₁, T₂, T₃ = 80, 160, 240 #30, 70, 140;
Tf = T₃ + 40;

p₁,p₂,p₃=7,10,5;
q₁,q₂,q₃=(p₁/T₁,p₂/T₁,p₃/T₁);

f₁(t) = tanh(q₁*t - p₁/2)
f₂(t) = tanh(q₂*t - p₂*(T₂/T₁ - 0.5))
f₃(t) = tanh(q₃*t - p₃*(T₃/T₁ - 0.5))

f_vec_1(t) = map(i->f₁(t)^i, 0:N+1) / 2
f_vec_2(t) = map(i->f₂(t)^i, 0:N+1) / 2
f_vec_3(t) = map(i->f₃(t)^i, 0:N+1) / 2


#=
using Plots

plot(tgrid, hcat(f_vec.(tgrid)...)')
=#

c11 = q₁*vcat([1, -0, -1], zeros(Int64,N-1))
c12 = q₂*vcat([1, -0, -1], zeros(Int64,N-1))
c13 = q₃*vcat([1, -0, -1], zeros(Int64,N-1))

kmax = length(c11)-2

ck1 = zeros(length(c11)+1, kmax)
ck1[1:end-1,1] = copy(c11)

ck2 = zeros(length(c12)+1, kmax)
ck2[1:end-1,1] = copy(c12)

ck3 = zeros(length(c13)+1, kmax)
ck3[1:end-1,1] = copy(c13)

for k=1:kmax-1
    for n=0:length(c11)-1
        ck1[n+1, k+1] = mapreduce(i-> (i+1)*ck1[i+2,k]*c11[n-i+1], +, 0:n)
        ck2[n+1, k+1] = mapreduce(i-> (i+1)*ck2[i+2,k]*c12[n-i+1], +, 0:n)
        ck3[n+1, k+1] = mapreduce(i-> (i+1)*ck3[i+2,k]*c13[n-i+1], +, 0:n)
    end
end


# Reference Trajectory
ref(t) = (1+f₁(t))/2 + (1+f₂(t))/2 - (1+f₃(t))

# Reference Derivatives 
function der_ref(t)
    f_vec_data_1 = vcat(f_vec_1(t),0)
    f_vec_data_2 = vcat(f_vec_2(t),0)
    f_vec_data_3 = -2*vcat(f_vec_3(t),0)

    res1 = ck1'*f_vec_data_1
    res2 = ck2'*f_vec_data_2
    res3 = ck3'*f_vec_data_3

    return res1 + res2 + res3
end



Tinv = inv(T)
q = hcat(-(C*A^N)*Tinv, 1) / (C*A^(N-1)*B);

function input_signal(t)
    r_vec = vcat(ref(t), der_ref(t))   
   return  q*r_vec 
end

heateq(z, p, t) = A*z + B*input_signal(t)

# Initial conditions
z0 = zeros(N)

# Solving the differential equation
tsave = 1;
tspan = (0, Tf)
using OrdinaryDiffEq
alg = Tsit5()
prob = ODEProblem(heateq, z0, tspan)
sol = solve(prob, alg, saveat=tsave)



using Plots
ts = 0.1
tgrid = 0 : ts : Tf

# Reference
plot(tgrid, ref.(tgrid))

# Reference Derivatives
plot(tgrid,hcat(der_ref.(tgrid)...)[1,:])

# Simulation of open-loop
plot(sol)
surface(Array(sol))

tgrid_sim = 0 : tsave : Tf;
plot(tgrid_sim, input_signal.(tgrid_sim))
ref_data_exp = hcat(tgrid_sim, ref.(tgrid_sim))
der_ref_data_exp = hcat(tgrid_sim, hcat(der_ref.(tgrid_sim)...)')
input_data_exp = hcat(tgrid_sim, input_signal.(tgrid_sim))
sol_data_exp = hcat(tgrid_sim, Array(sol)')


# Store results
using DelimitedFiles
path2folder = "results/data/"
ref_fname = "reference_data.txt"
der_ref_fname = "derivative_ref_data.txt"
input_fname = "input_data.txt"
sol_fname = "heat_eq_sim_data.txt"

path2file = path2folder * ref_fname
open(path2file, "w") do io
    writedlm(io, ref_data_exp, ' ')
end;

path2file = path2folder * der_ref_fname
open(path2file, "w") do io
    writedlm(io, der_ref_data_exp, ' ')
end;


path2file = path2folder * input_fname
open(path2file, "w") do io
    writedlm(io, input_data_exp, ' ')
end;


path2file = path2folder * sol_fname
open(path2file, "w") do io
    writedlm(io, sol_data_exp, ' ')
end;

