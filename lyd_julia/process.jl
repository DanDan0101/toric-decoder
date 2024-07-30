using Printf
using DelimitedFiles

list_L = [100, 200, 500, 1000, 2000]
list_PANY = [0.04]
list_ETA  = [0.60]
list_BETA  = 0.60:0.20:1.40

sizes = [0 1 2 3 4]# 5 6 7 8 9]


dir_input = "./data/"
dir_output = "./data_avg/"

size2 = 12

for L in list_L, P in list_PANY, Q in list_ETA, β in list_BETA
	DEPTH = 16*L+1
	#sizeB = 8

	run_L = sizes[1]*1000 #(s*100)*10
	run_R = sizes[end]*1000 + 999 #(s*100+99)*10 + 9
	# run_R = sizes[1]*1000+999

	count_SA = 0
	count_SB = 0
    count_SC = 0
	SA  = zeros(DEPTH, 1)
	# SB  = zeros(DEPTH, L)
    # SC  = zeros(DEPTH, L)
	# SA2 = zeros(DEPTH, L+1)
	# SA3 = zeros(DEPTH, L+1)
	# SA3 = zeros(DEPTH, div(L,2))
	# DA  = zeros(DEPTH, L)

	for run = run_L:run_R
#=
		if k == size2
			println(run)
		end
=#

#		@show run

		filename = string(dir_input, "SA_L_", lpad(L, 5, '0'), "_PANY_", @sprintf("%.3f", P), "_ETA_", @sprintf("%.3f", Q), "_BETA_", @sprintf("%.3f", β), "_RUN_", lpad(run, 4, '0'), ".csv")
		if isfile(filename)
			data_tmp = readdlm(filename, ' ')
			# @show size(data_tmp)
			# @show size(SA)
			if size(data_tmp) == size(SA)
				count_SA += 1
				SA += data_tmp
#				SA2 += data_tmp.^2
#				SA3 += data_tmp.^3

				#histogram ... 

				# for t = 1:DEPTH, l in 1:div(L,2)
				# 	SA3[t, l] += data_tmp[t,l]*data_tmp[t,2*l]
				# end
			end
		end

		# filename = string(dir_input, "_L_",  L, "_SB", "_P_", @sprintf("%.4f", P), "_Q_", @sprintf("%.4f", Q), "_RUN_", lpad(run, 4, 0), ".csv")
		# if isfile(filename)
		# 	data_tmp = readdlm(filename, ' ')
		# 	# @show size(data_tmp)
		# 	# @show size(SA)
		# 	if size(data_tmp) == size(SB)
		# 		count_SB += 1
		# 		SB += data_tmp
		# 		# SA2 += data_tmp.^2

		# 		# for t = 1:DEPTH, l in 1:div(L,2)
		# 		# 	SA3[t, l] += data_tmp[t,l]*data_tmp[t,2*l]
		# 		# end
		# 	end
		# end
        
        
        # filename = string(dir_input, "_L_",  L, "_SC", "_P_", @sprintf("%.4f", P), "_Q_", @sprintf("%.4f", Q), "_RUN_", lpad(run, 4, 0), ".csv")
		# if isfile(filename)
		# 	data_tmp = readdlm(filename, ' ')
		# 	# @show size(data_tmp)
		# 	# @show size(SA)
		# 	if size(data_tmp) == size(SC)
		# 		count_SC += 1
		# 		SC += data_tmp
		# 		# SA2 += data_tmp.^2

		# 		# for t = 1:DEPTH, l in 1:div(L,2)
		# 		# 	SA3[t, l] += data_tmp[t,l]*data_tmp[t,2*l]
		# 		# end
		# 	end
		# end

		# filename = string(dir_input, "_L_",  L, "_DA", "_P_", @sprintf("%.4f", P), "_Q_", @sprintf("%.4f", Q), "_RUN_", lpad(run, 4, 0), ".csv")
		# if isfile(filename)
		# 	data_tmp = readdlm(filename, ' ')
		# 	#@show data_tmp
		# 	#@show size(data_tmp)
		# 	#@show size(SA[k])
		# 	if size(data_tmp) == size(DA)
		# 		count_DA += 1
		# 		DA += data_tmp
		# 	end
		# end

#=
		filename = string(dir_input, "_N_",  n, "_MI_", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h), "_RUN_", lpad(run, 4, 0), ".csv")
		if isfile(filename)
			data_tmp = readdlm(filename, ' ')
			#@show data_tmp
			#@show size(data_tmp)
			#@show size(SA[k])
			if size(data_tmp) == size(MI[k])
				count_MI[k] += 1
				MI[k] += data_tmp
			end
		end
=#

#=
		filename = string(dir_input, "_N_",  n, "_DA_", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h), "_RUN_", lpad(run, 4, 0), ".csv")
		if isfile(filename)
			data_tmp = readdlm(filename, ' ')
			#@show data_tmp
			#@show size(data_tmp)
			#@show size(SA[k])
			if size(data_tmp) == size(DA[k])
				count_DA[k] += 1
				DA[k] += data_tmp
			end
		end
=#
#=
		filename = string(dir_input, "_N_",  n, "_DB_", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h), "_RUN_", lpad(run, 4, 0), ".csv")
		if isfile(filename)
			data_tmp = readdlm(filename, ' ')
			#@show data_tmp
			#@show size(data_tmp)
			#@show size(SA[k])
			if size(data_tmp) == size(DB[k])
				count_DB[k] += 1
				DB[k] += data_tmp
			end
		end
=#
#=
		filename = string(dir_input, "_N_",  n, "_SB", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h), "_RUN_", lpad(run, 4, 0), ".csv")
		if isfile(filename)
			data_tmp = readdlm(filename, ' ')
			if size(data_tmp) == size(SB[k])
				count_SB[k] += 1
				SB[k] += data_tmp
			end
		end

		filename = string(dir_input, "_N_",  n, "_SAB", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h), "_RUN_", lpad(run, 4, 0), ".csv")
		if isfile(filename)
			data_tmp = readdlm(filename, ' ')
			if size(data_tmp) == size(SAB[k])
				count_SAB[k] += 1
				SAB[k] += data_tmp
			end
		end
=#
	end

		
		if count_SA > 0
			@show count_SA
			filename = string(dir_output, "SA_L_", lpad(L, 5, '0'), "_PANY_", @sprintf("%.3f", P), "_ETA_", @sprintf("%.3f", Q), "_BETA_", @sprintf("%.3f", β), ".csv")
			writedlm(filename, SA / count_SA, " ")
			filename = string(dir_output, "_L_",  L, "_SA2", "_P_", @sprintf("%.4f", P), "_Q_", @sprintf("%.4f", Q), ".csv")
			# writedlm(filename, SA2 / count_SA, " ")
			filename = string(dir_output, "_L_",  L, "_SA3", "_P_", @sprintf("%.4f", P), "_Q_", @sprintf("%.4f", Q), ".csv")
			# writedlm(filename, SA3 / count_SA, " ")
		end

		# if count_SB > 0
		# 	#@show count_SA
		# 	filename = string(dir_output, "_L_",  L, "_SB", "_P_", @sprintf("%.4f", P), "_Q_", @sprintf("%.4f", Q), ".csv")
		# 	writedlm(filename, SB / count_SB, " ")
		# 	# filename = string(dir_output, "_L_",  L, "_SA2", "_P_", @sprintf("%.4f", P), "_Q_", @sprintf("%.4f", Q), ".csv")
		# 	# writedlm(filename, SA2 / count_SA, " ")
		# 	# filename = string(dir_output, "_L_",  L, "_SA3", "_P_", @sprintf("%.4f", P), "_Q_", @sprintf("%.4f", Q), ".csv")
		# 	# writedlm(filename, SA3 / count_SA, " ")
		# end
    
        # if count_SC > 0
		# 	#@show count_SA
		# 	filename = string(dir_output, "_L_",  L, "_SC", "_P_", @sprintf("%.4f", P), "_Q_", @sprintf("%.4f", Q), ".csv")
		# 	writedlm(filename, SC / count_SC, " ")
		# 	# filename = string(dir_output, "_L_",  L, "_SA2", "_P_", @sprintf("%.4f", P), "_Q_", @sprintf("%.4f", Q), ".csv")
		# 	# writedlm(filename, SA2 / count_SA, " ")
		# 	# filename = string(dir_output, "_L_",  L, "_SA3", "_P_", @sprintf("%.4f", P), "_Q_", @sprintf("%.4f", Q), ".csv")
		# 	# writedlm(filename, SA3 / count_SA, " ")
		# end

		# if count_DA > 0
		# 	@show count_DA
		# 	filename = string(dir_output, "_L_",  L, "_DA", "_P_", @sprintf("%.4f", P), "_Q_", @sprintf("%.4f", Q), ".csv")
		# 	writedlm(filename, DA / count_DA, " ")
		# end
#=
		if count_SA[k] > 0
			#filename = string(dir_output, "_N_",  n, "_SA_", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h),  "_RUN_", lpad(run_L, 4, 0), ".csv")
			filename = string(dir_output, "_N_",  n, "_MI_", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h), ".csv")
			writedlm(filename, MI[k] / count_MI[k], " ")
		end
		=#
#=
		if count_DA[k] > 0
			#filename = string(dir_output, "_N_",  n, "_SA_", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h),  "_RUN_", lpad(run_L, 4, 0), ".csv")
			filename = string(dir_output, "_N_",  n, "_DA_", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h), ".csv")
			writedlm(filename, DA[k] / count_DA[k], " ")
		end

		if count_DB[k] > 0
			#filename = string(dir_output, "_N_",  n, "_SA_", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h),  "_RUN_", lpad(run_L, 4, 0), ".csv")
			filename = string(dir_output, "_N_",  n, "_DB_", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h), ".csv")
			writedlm(filename, DB[k] / count_DB[k], " ")
		end
=#
#=
		if count_SB[k] > 0
			filename = string(dir_output, "_N_",  n, "_SB_", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h),  "_RUN_", lpad(run_L, 4, 0), ".csv")
			writedlm(filename, SB[k] / count_SB[k], " ")
		end

		if count_SAB[k] > 0
			filename = string(dir_output, "_N_",  n, "_SAB_", lpad(k, 2, 0), "_P_", @sprintf("%.4f", h), "_RUN_", lpad(run_L, 4, 0), ".csv")
			writedlm(filename, SAB[k] / count_SAB[k], " ")
		end
=#

end
