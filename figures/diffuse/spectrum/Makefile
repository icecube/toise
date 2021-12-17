

gen2-radio_unfolding.double-powerlaw.json.gz:
	toise-figure-data toise.figures.diffuse.spectrum.unfold --astro 2.3 --gamma -2.5 --gamma-step 0 --steps 100 -d IceCube 15 -d Gen2-InIce 15 -d Gen2-Radio 15 -o $@

gen2-radio_unfolding.double-powerlaw.pdf: gen2-radio_unfolding.double-powerlaw.json.gz
	toise-plot diffuse.spectrum.unfolded_flux $^ -o $@

