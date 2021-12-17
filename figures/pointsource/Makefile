
flare_significance_energyrange.pdf: flare_significance_icecube_10yr_energyrange.json.gz flare_significance_gen2_10yr_energyrange.json.gz
	toise-plot pointsource.flare.single_flare_time_to_signficance $^ -o $@

flare_significance_energyrange.json: flare_significance_icecube_10yr_energyrange.json.gz flare_significance_gen2_10yr_energyrange.json.gz
	toise-plot pointsource.flare.single_flare_time_to_signficance flare_significance_icecube_10yr_energyrange.json.gz flare_significance_gen2_10yr_energyrange.json.gz -d flare_significance_energyrange.json

flare_significance_icecube_10yr_energyrange.json.gz:
	toise-figure-data pointsource.flare.single_flare_time_to_signficance -d IceCube-TracksOnly 10 --emin 3e4 --emax 3e6 -o flare_significance_icecube_10yr_energyrange

flare_significance_gen2_10yr_energyrange.json.gz:
	toise-figure-data pointsource.flare.single_flare_time_to_signficance -d Gen2-InIce-TracksOnly 10 --emin 3e4 --emax 3e6 -o flare_significance_gen2_10yr_energyrange

