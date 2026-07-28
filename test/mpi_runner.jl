using MPI: mpiexec

# A cache mismatch hangs rather than errors, because the constructor's receives are
# blocking — so every run gets a hard wall-clock kill.
function run_mpi_worker(case::String, nranks::Int; timeout = 180)
    worker = joinpath(@__DIR__, "mpi_beliefpropagation_worker.jl")
    project = Base.active_project()
    cmd = `$(mpiexec()) -n $nranks $(Base.julia_cmd()) --project=$project $worker $case`
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
