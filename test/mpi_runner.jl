using MPI: mpiexec

# Launching mpiexec dominates the runtime: each rank loads the package and JIT-compiles the BP
# paths from scratch, which costs far more than any single case. So pass every case of a given
# rank count in one call and let the worker loop over them in-process.
#
# A cache mismatch hangs rather than errors, because the exchange's receives are blocking — so
# every run gets a hard wall-clock kill.
function run_mpi_worker(cases::Vector{String}, nranks::Int; timeout = 180 * length(cases))
    worker = joinpath(@__DIR__, "mpi_beliefpropagation_worker.jl")
    project = Base.active_project()
    # --startup-file=no: a user startup.jl loading packages absent from the package environment
    # would fail every rank before the test runs.
    cmd = `$(mpiexec()) -n $nranks $(Base.julia_cmd()) --startup-file=no --project=$project $worker $cases`
    # stdio must be named explicitly: run(...; wait = false) sends it to devnull.
    p = run(pipeline(ignorestatus(cmd); stdout, stderr); wait = false)
    timer = Timer(_ -> process_running(p) && kill(p, Base.SIGKILL), timeout)
    try
        wait(p)
    finally
        close(timer)
    end
    return success(p)
end

run_mpi_worker(case::String, nranks::Int; kwargs...) =
    run_mpi_worker([case], nranks; kwargs...)
