# scripts used to create/initiate the project 

project_name = "model_agent"

if false
    using Pkg
    # to do it manually with Pkg
    Pkg.generate("projects/$project_name")
    Pkg.instantiate()
    Pkg.activate("projects/$project_name")  
    cd("projects/$project_name")
end

 
# or via DrWatson
using DrWatson

initialize_project("projects/$project_name"; 
    authors="Jae S. Choi",
    template = [
        "archive",
        "data",  
        "docs",
        "ignore",
        "outputs",
        "reference",
        "scripts",
        "src",
        "test"
    ]
)

quickactivate( joinpath(homedir(), "projects", project_name ) )
  
cd( projectdir() )
 
