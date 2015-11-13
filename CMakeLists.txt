
i3_project(gen2_analysis PYTHON_DIR python)

execute_process(COMMAND rsync -avz --cvs-exclude
      "data.icecube.wisc.edu:/data/user/jvansanten/projects/2015/gen2_benchmark/data/"
      "${PROJECT_SOURCE_DIR}/resources/data/")
