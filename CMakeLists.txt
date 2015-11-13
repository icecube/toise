
i3_project(gen2_benchmark PYTHON_DIR python)

execute_process(rsync -avz
      "data.icecube.wisc.edu:/data/user/jvansanten/projects/2015/gen2_benchmark/data/"
      "${PROJECT_SOURCE_DIR}/resources/data/")
