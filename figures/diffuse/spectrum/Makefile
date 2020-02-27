

gen2-radio_unfolding.double-powerlaw.json.gz:
	gen2-figure-data gen2_analysis.figures.diffuse.spectrum.unfold --astro 2.3 --gamma -2.5 --gamma-step 0 --steps 100 -d IceCube 15 -d Gen2-InIce 15 -d Gen2-Radio 15 -o $@

gen2-radio_unfolding.double-powerlaw.pdf: gen2-radio_unfolding.double-powerlaw.json.gz
	gen2-plot diffuse.spectrum.unfolded_flux $^ -o $@

