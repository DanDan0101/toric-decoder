using DelimitedFiles
# for I/O to files. consider using JLD2 for better performance

using Printf
# for string manipulations

# using LinearAlgebra


# Statistics: finally, pymatching

mutable struct state
	L :: Int64	# size of the lattice
	N :: Int64	# number of anyons
	ρ :: Matrix{Int64} 	 # onsite anyon parity
	ϕ :: Matrix{Float64} # field configuration
end

function next(i, L)
	return i%L + 1
end

function prev(i, L)
	return (i-1+L-1)%L + 1
end


function init(L, p_anyon)

	N = 0

	ρ = Matrix{Int64}(undef, L, L)
	for i in 1:L, j in 1:L
		ρ[i,j] = 0
	end

	for i in 1:L, j in 1:L
		if rand() < p_anyon
			ρ[i,j] = (ρ[i,j]+1)%2
			ρ[i,next(j,L)] = (ρ[i,next(j,L)]+1)%2
		end

		if rand() < p_anyon
			ρ[i,j] = (ρ[i,j]+1)%2
			ρ[next(i,L),j] = (ρ[next(i,L),j]+1)%2
		end
	end

	for i in 1:L, j in 1:L
		if ρ[i,j] > 0
			N += 1
		end
	end

	ϕ = Matrix{Float64}(undef, L, L)
	for i in 1:L, j in 1:L
		ϕ[i,j] = 0.0
	end

	return state(L, N, ρ, ϕ)
end


function init_two_particle(L, p_anyon)


	ρ = Matrix{Int64}(undef, L, L)
	for i in 1:L, j in 1:L
		ρ[i,j] = 0
	end

	N = 2
	# ρ[div(L,4), div(L, 4)] = 1
	# ρ[3*div(L,4), 3*div(L, 4)] = 1

	ρ[10, 10] = 1
	ρ[18, 18] = 1

	ϕ = Matrix{Float64}(undef, L, L)
	for i in 1:L, j in 1:L
		ϕ[i,j] = 0.0
	end

	return state(L, N, ρ, ϕ)
end


function init_superlattice(L, spacing)

	N = 0

	ρ = Matrix{Int64}(undef, L, L)
	for i in 1:L, j in 1:L
		if (i%spacing == 0) && (j%spacing == 0)
			ρ[i,j] = 1
			N += 1
		else
			ρ[i,j] = 0
		end
	end
	
	# ρ[div(L,4), div(L, 4)] = 1
	# ρ[3*div(L,4), 3*div(L, 4)] = 1


	ϕ = Matrix{Float64}(undef, L, L)
	for i in 1:L, j in 1:L
		ϕ[i,j] = 0.0
	end

	return state(L, N, ρ, ϕ)
end


function print_anyon(s::state)
	for i in 1:s.L
		for j in 1:s.L
			if s.ρ[i,j] == 0
				print('.')
			else
				print('*')
			end
		end

		print('\n')
	end

	print("\n\n\n")
end

function update_field(s::state, η)
	ϕ0 = copy(s.ϕ)

	for i in 1:s.L, j in 1:s.L
		s.ϕ[i,j] = (1-η)*ϕ0[i,j] + η * (0.25 * (ϕ0[prev(i,s.L),j] + ϕ0[next(i,s.L),j] + ϕ0[i,prev(j,s.L)] + ϕ0[i,next(j,s.L)]) + s.ρ[i,j] )
	end
end

function update_anyon(s::state, β)
	ρ0 = copy(s.ρ)
	
	for i in 1:s.L, j in 1:s.L

		if ρ0[i,j] > 0

			# β = 1.2

			new_position = []

			push!( new_position, (i,j,1) )
			# @show i,j,s.ϕ[i,j]
			push!( new_position, (prev(i,s.L),j,exp(β*(s.ϕ[prev(i,s.L),j]-s.ϕ[i,j]) ) ) )
			# @show prev(i,s.L),j,s.ϕ[prev(i,s.L),j]
			push!( new_position, (next(i,s.L),j,exp(β*(s.ϕ[next(i,s.L),j]-s.ϕ[i,j]) ) ) )
			# @show next(i,s.L),j,s.ϕ[next(i,s.L),j]
			push!( new_position, (i,prev(j,s.L),exp(β*(s.ϕ[i,prev(j,s.L)]-s.ϕ[i,j]) ) ) )
			# @show i,prev(j,s.L),s.ϕ[i,prev(j,s.L)]
			push!( new_position, (i,next(j,s.L),exp(β*(s.ϕ[i,next(j,s.L)]-s.ϕ[i,j]) ) ) )
			# @show i,next(j,s.L),s.ϕ[i,next(j,s.L)]

			# @show new_position

			Ztotal = sum([new_position[iter][3] for iter in 1:length(new_position)])

			ii, jj = i, j

			rr = rand()
			if rr < new_position[1][3] / Ztotal
				ii, jj = new_position[1][1], new_position[1][2]
			elseif rr < (new_position[1][3]+new_position[2][3]) / Ztotal
				ii, jj = new_position[2][1], new_position[2][2]
			elseif rr < (new_position[1][3]+new_position[2][3]+new_position[3][3]) / Ztotal
				ii, jj = new_position[3][1], new_position[3][2]
			elseif rr < (new_position[1][3]+new_position[2][3]+new_position[3][3]+new_position[4][3]) / Ztotal
				ii, jj = new_position[4][1], new_position[4][2]
			elseif rr < (new_position[1][3]+new_position[2][3]+new_position[3][3]+new_position[4][3]+new_position[5][3]) / Ztotal
				ii, jj = new_position[5][1], new_position[5][2]
			end

			s.ρ[i,j] -= 1
			s.ρ[ii,jj] += 1

			# @assert ρ0[i,j] == 1

			# new_position = [(prev(i,s.L),j)]

			# if s.ϕ[next(i,s.L),j] > s.ϕ[new_position[1][1], new_position[1][2]]
			# 	new_position = [(next(i,s.L),j)]
			# elseif s.ϕ[next(i,s.L),j] == s.ϕ[new_position[1][1], new_position[1][2]]
			# 	push!(new_position, (next(i,s.L),j))
			# end

			# if s.ϕ[i,prev(j,s.L)] > s.ϕ[new_position[1][1], new_position[1][2]]
			# 	new_position = [(i,prev(j,s.L))]
			# elseif s.ϕ[i,prev(j,s.L)] == s.ϕ[new_position[1][1], new_position[1][2]]
			# 	push!(new_position, (i,prev(j,s.L)))
			# end

			# if s.ϕ[i,next(j,s.L)] > s.ϕ[new_position[1][1], new_position[1][2]]
			# 	new_position = [(i,next(j,s.L))]
			# elseif s.ϕ[i,next(j,s.L)] == s.ϕ[new_position[1][1], new_position[1][2]]
			# 	push!(new_position, (i,next(j,s.L)))
			# end

			# if rand() < 0.8
			# 	s.ρ[i,j] -= 1
			# 	ii, jj = new_position[rand(1:length(new_position))]
			# 	s.ρ[ii,jj] += 1
			# end
		end
	end

	for i in 1:s.L, j in 1:s.L
		if s.ρ[i,j] > 0
			s.N -= 2*div(s.ρ[i,j], 2)
			s.ρ[i,j] = mod(s.ρ[i,j], 2)
		end
	end
end

function main(L, T, p_anyon, η, β, run)

	# s = init(L, p_anyon)
	s = init_superlattice(L, floor(Int, 1.0/sqrt(p_anyon)))
	# s = init_two_particle(L, p_anyon)

	# print_anyon(s)
	# print(s.N / (s.L^2), '\n')

	SA = [s.N]


	for t in 1:20*T
		for tt in 1:1
			update_field(s, 1.0)
		end
	end

	for t in 1:T

		for tt in 1:1
			update_field(s, η)
		end

		# @show s.ϕ

		for tt in 1:1
			update_anyon(s, β)
		end

		# print_anyon(s)

		push!(SA, s.N)
	end
	
	# print_anyon(s)
	# print(s.N / (s.L^2), '\n')

	writedlm(string("./data/SA_L_", lpad(L, 5, '0'), "_PANY_", @sprintf("%.3f", p_anyon), "_ETA_", @sprintf("%.3f", η), "_BETA_", @sprintf("%.3f", β), "_RUN_", lpad(run, 4, '0'), ".csv"), SA, ' ')

	return 0
end