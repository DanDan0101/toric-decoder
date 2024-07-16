include("main.jl")

L       = parse(Int64,	ARGS[1])
PANY    = parse(Float64,	ARGS[2])
ETA		= parse(Float64,	ARGS[3])
BETA	= parse(Float64,	ARGS[4])
run     = parse(Int64,	ARGS[5])

main(L, 16*L, PANY, ETA, BETA, run)
# main(L, 20, PANY, ETA, run)

# julia run.jl 100 0.01 0.50 0