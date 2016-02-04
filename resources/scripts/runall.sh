#!/bin/zsh

detectors=("--geometry IceCube --veto-area 0"                            \
"--geometry IceCube --veto-area 25 --veto-threshold 1e5"                 \
"--geometry Sunflower --spacing 240 --veto-area 0 --veto-threshold 1e5"  \
"--geometry Sunflower --spacing 240 --veto-area 25 --veto-threshold 1e5" \
#"--geometry Sunflower --spacing 240 --veto-area 25 --veto-threshold 1e5" \
"--geometry Sunflower --spacing 240 --veto-area 75 --veto-threshold 1e5" \
"--geometry Sunflower --spacing 200 --veto-area 75 --veto-threshold 1e5" \
#"--geometry Sunflower --spacing 240 --veto-area 75 --veto-threshold 1e5" \
"--geometry Sunflower --spacing 300 --veto-area 75 --veto-threshold 1e5" \
"--geometry EdgeWeighted --spacing 240 --veto-area 75 --veto-threshold 1e5" \
)

figures=(
"survey_volume"                  \
"survey_volume --angular-resolution-scale=0.5"                  \
#"grb"                                     \
#"diffuse_index --energy-threshold 1e5"    \
#"diffuse_index --energy-threshold 1e6"    \
#"galactic_diffuse --energy-threshold 1e4" \
#"galactic_diffuse --energy-threshold 1e5" \
#"gzk"                                     \
)

. ~/.virtualenv/standard/bin/activate 

for f in $figures; do
	for d in $detectors; do
		# echo python calculate_fom.py $d $f
		eval python calculate_fom.py $d $f
	done
	# break
done
