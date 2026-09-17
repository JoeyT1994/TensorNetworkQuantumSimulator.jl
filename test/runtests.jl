using Test
using Distributed   # top level: the `@everywhere` below must be defined when the block is lowered

# The test files are independent (each wraps itself in an anonymous module), so they run in PARALLEL
# worker processes by default: the suite is ~65 min serial and ~16 min as the longest single file
# (measured 2026-09-17, test_ctmenvironment.jl). `TNQS_TEST_WORKERS=n` sets the worker count (default:
# one per file, capped at the machine's threads ÷ 4 and at 6); `TNQS_TEST_WORKERS=0` or `1` runs the
# files serially in this process, in the order below. `TNQS_TEST_FILES=a,b` (basenames without `test_`
# and `.jl`, e.g. `ctmenvironment,dmrg`) selects a subset. The package must already be precompiled for
# the workers; `TNQS_SKIP_PRECOMPILE_WORKLOAD=1` in the environment skips the PrecompileTools workload
# (150 s per source edit → ~20 s) and is inherited by the workers.
const TEST_FILES = [
    "test_constructors.jl",
    "test_forms.jl",
    "test_expect.jl",
    "test_boundarymps.jl",
    "test_ctmenvironment.jl",
    "test_beliefpropagation.jl",
    "test_apply.jl",
    "test_sampling.jl",
    "test_truncate.jl",
    "test_contraction_sequences.jl",
    "test_tensors.jl",
    "test_dmrg.jl",
    "test_gpu_paths.jl",
]

function selected_files()
    raw = strip(get(ENV, "TNQS_TEST_FILES", ""))
    isempty(raw) && return TEST_FILES
    want = ["test_$(strip(x)).jl" for x in split(raw, ',') if !isempty(strip(x))]
    unknown = setdiff(want, TEST_FILES)
    isempty(unknown) || error("TNQS_TEST_FILES: unknown test file(s) $(join(unknown, ", ")); known: $(join(TEST_FILES, ", "))")
    return filter(∈(want), TEST_FILES)
end

files = selected_files()
# Longest files first so the stragglers start early: ctmenvironment (~16 min), tensors (~15), dmrg (~12.5).
const LONG_FIRST = ["test_ctmenvironment.jl", "test_tensors.jl", "test_dmrg.jl", "test_gpu_paths.jl"]
sort!(files; by = f -> (something(findfirst(==(f), LONG_FIRST), length(LONG_FIRST) + 1), f))

default_workers = min(length(files), max(1, Sys.CPU_THREADS ÷ 4), 6)
nworkers_wanted = parse(Int, get(ENV, "TNQS_TEST_WORKERS", string(default_workers)))

if nworkers_wanted <= 1 || length(files) <= 1
    using TensorNetworkQuantumSimulator
    @testset "TensorNetworkQuantumSimulator.jl" begin
        for f in files
            include(f)
        end
    end
else
    # Same project, same environment variables; each worker loads the (precompiled) package once.
    procs_added = addprocs(nworkers_wanted; exeflags = "--project=$(Base.active_project())")
    try
        @everywhere procs_added begin
            using Test
            using TensorNetworkQuantumSimulator
        end
        testdir = @__DIR__
        @info "Running $(length(files)) test files on $(length(procs_added)) workers" files
        # Each file runs in its own top-level testset on the worker; the summary and any failure are
        # collected here, so a failing file fails the suite with its own report.
        results = pmap(files) do f
            t0 = time()
            # a top-level testset with failures throws `TestSetException` when it finishes (after
            # printing its own report on the worker); catch it so every file still gets to run
            failed = try
                @testset "$f" begin
                    include(joinpath(testdir, f))
                end
                false
            catch err
                err isa Test.TestSetException || rethrow()
                true
            end
            return (; file = f, seconds = round(time() - t0; digits = 1), failed)
        end
        @testset "TensorNetworkQuantumSimulator.jl" begin
            for r in sort(results; by = r -> -r.seconds)
                @info "$(r.file): $(r.seconds) s" failed = r.failed
                @test !r.failed
            end
        end
    finally
        rmprocs(procs_added)
    end
end
