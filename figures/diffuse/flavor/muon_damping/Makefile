
icecube+gen2-inice.json.gz:
	toise-figure-data diffuse.flavor.muon_damping_constraints -d IceCube 12 Gen2-InIce 10 --steps 100 -o $@

muon_damped_flavor.pdf: icecube+gen2-inice.json.gz
	toise-plot diffuse.flavor.muon_damping $^ -o $@

