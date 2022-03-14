
muon_damped_flavor.pdf: icecube+gen2-inice.json.gz
	toise-plot diffuse.flavor.muon_damping $^ -o $@

icecube+gen2-inice.json.gz:
	toise-figure-data diffuse.flavor.muon_damping_constraints --clean -d IceCube 15 Gen2-InIce 10 --steps 100 -o $@

.PHONY: clean
clean:
	-rm icecube+gen2-inice.json.gz muon_damped_flavor.pdf

