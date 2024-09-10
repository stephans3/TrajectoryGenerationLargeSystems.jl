using TrajectoryGenerationLargeSystems
using Documenter

DocMeta.setdocmeta!(TrajectoryGenerationLargeSystems, :DocTestSetup, :(using TrajectoryGenerationLargeSystems); recursive=true)

makedocs(;
    modules=[TrajectoryGenerationLargeSystems],
    authors="Stephan Scholz",
    sitename="TrajectoryGenerationLargeSystems.jl",
    format=Documenter.HTML(;
        canonical="https://stephans3.github.io/TrajectoryGenerationLargeSystems.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/stephans3/TrajectoryGenerationLargeSystems.jl",
    devbranch="main",
)
