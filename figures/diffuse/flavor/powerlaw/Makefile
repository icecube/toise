
icecube_8yr.json.gz:
	toise-figure-data diffuse.flavor.confidence_levels -d IceCube 8 -o $@

icecube_15yr.json.gz:
	toise-figure-data diffuse.flavor.confidence_levels -d IceCube 15 -o $@

icecube+gen2-inice.json.gz:
	toise-figure-data diffuse.flavor.confidence_levels -d IceCube 15 Gen2-InIce 10 -o $@

flavor_triangle.pdf: icecube_8yr.json.gz icecube_15yr.json.gz icecube+gen2-inice.json.gz
	toise-plot diffuse.flavor.triangle $^ -o $@

