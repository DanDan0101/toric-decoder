using DelimitedFiles
using Printf

list_L = [100, 200, 500, 1000]
list_PANY = [0.04]
list_ETA  = [0.60]
list_BETA  = 0.60:0.20:1.40

batch = 000:009

dir = "./script/"
data = "0721_CA_grav_field_hold_release"

dozen = 10
ncore = 5

for L in list_L, P in list_PANY, Q in list_ETA, β in list_BETA, b in batch
#for h in hh, s in sizes, b in batch
	run_L = (b) * dozen
	run_R = (b) * dozen + (dozen-1)
	filename = string(dir, data, "_L_", L, "_PANY_", @sprintf("%.4f", P),  "_ETA_", @sprintf("%.4f", Q), "_BETA_", @sprintf("%.4f", β), "_", lpad(run_L, 4, '0') )

	println("sbatch ", filename)
	#println("sbatch -q short ", filename)
	open(filename, "w") do file
		write(file, "#!/bin/bash -l\n")
		write(file, "#SBATCH --nodes=1 --ntasks-per-node=$ncore\n")
		#write(file, "#SBATCH --time=1:00:00\n")
		if (b == batch[end] && P == list_PANY[end] && Q == list_ETA[end])
			write(file, "#SBATCH --mail-user=liyd@stanford.edu\n")
			write(file, "#SBATCH --mail-type=start,end\n")
		end
		write(file, "cd \$SLURM_SUBMIT_DIR\n")


		write(file, "for i in {$run_L..$run_R} \n")
		write(file, "do\n")
		write(file, "    echo \"\$i\"\n")
		write(file, "    julia run.jl $L $P $Q $β \$i& \n")
		write(file, "done\n")
		write(file, "wait\n")

		#write(file, "julia process_10.jl $run_L& \n")

		write(file, "echo \"finish\"\n")
	end
end
