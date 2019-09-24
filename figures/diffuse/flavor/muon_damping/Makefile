
icecube+gen2-inice.json.gz:
	gen2-figure-data diffuse.flavor.muon_damping_constraints -d IceCube 15 Gen2-InIce 10 --steps 100 -o $@

muon_damped_flavor.pdf: icecube+gen2-inice.json.gz
	gen2-plot diffuse.flavor.muon_damping $^ -o $@

