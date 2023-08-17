
flare_significance_energyrange.pdf: flare_significance_icecube_10yr_energyrange.json.gz flare_significance_gen2_10yr_energyrange.json.gz
	toise-plot pointsource.flare.single_flare_time_to_signficance $^ -o $@

flare_significance_energyrange.json: flare_significance_icecube_10yr_energyrange.json.gz flare_significance_gen2_10yr_energyrange.json.gz
	toise-plot pointsource.flare.single_flare_time_to_signficance flare_significance_icecube_10yr_energyrange.json.gz flare_significance_gen2_10yr_energyrange.json.gz -d flare_significance_energyrange.json

flare_significance_icecube_10yr_energyrange.json.gz:
	toise-figure-data pointsource.flare.single_flare_time_to_signficance -d IceCube-TracksOnly 10 --emin 3e4 --emax 3e6 -o flare_significance_icecube_10yr_energyrange

flare_significance_gen2_10yr_energyrange.json.gz:
	toise-figure-data pointsource.flare.single_flare_time_to_signficance -d Gen2-InIce-TracksOnly 10 --emin 3e4 --emax 3e6 -o flare_significance_gen2_10yr_energyrange

# 5.010^-11 (E/TeV)*-3.2 in TXS paper units = 1.99e-17 (TeV^-1 cm^-2 s^-2 at 100 TeV) 
time_to_significance_icecube_energyrange.json.gz:
	toise-figure-data pointsource.flare.single_flare_time_to_signficance -d IceCube-TracksOnly 1 \
		--dec 0 \
		--emin 1.5e3 --emax 1.5e4 \
		--flux 1.99e-17 --gamma -3.2 \
		--days 365 730 1095 1460 1825 2190 2555 2920 3285 3650 4015 4380 4745 5110 \
		--no-duration-penalty \
		-o time_to_significance_icecube_energyrange

time_to_significance_gen2_energyrange.json.gz:
	toise-figure-data pointsource.flare.single_flare_time_to_signficance -d Gen2-InIce-TracksOnly 1 \
		--dec 0 \
		--emin 1.5e3 --emax 1.5e4 \
		--flux 1.99e-17 --gamma -3.2 \
		--days 365 730 1095 1460 1825 2190 2555 2920 3285 3650 4015 4380 4745 5110 \
		--no-duration-penalty \
		-o time_to_significance_gen2_energyrange

time_to_significance_energyrange.pdf: time_to_significance_icecube_energyrange.json.gz time_to_significance_gen2_energyrange.json.gz
	toise-plot pointsource.flare.single_flare_time_to_signficance $^ -o $@

#DETECTORS = Gen2-InIce-TracksOnly-NoVeto "Gen2-InIce-TracksOnly-NoVeto Gen2-Radio" Gen2-InIce-NoVeto "Gen2-InIce-NoVeto Gen2-Radio" IceCube IceCube-TracksOnly Gen2-InIce Gen2-InIce-TracksOnly Gen2-Radio "Gen2-InIce Gen2-Radio"
DETECTORS = "Gen2-InIce-NoVeto Gen2-Radio" IceCube Gen2-InIce Gen2-InIce-NoVeto Gen2-Radio "Gen2-InIce Gen2-Radio"
# Flare sensitivity timescales: 100s, 500s, 1000s,14d, 100d and 10y

flare_sensitivity_100seconds.json.gz:
	years=$$(printf "%.7f" $$(echo "100/3600/24/365" | bc -l)); \
    toise-figure-data pointsource.flare.sensitivity $$(for d in $(DETECTORS); do echo -d; for element in $$d; do echo $$element $$years; done; done) $$(for d in $(DETECTORS); do echo -o $(basename $(basename $@))_$$(echo $$d | sed -e 's| |+|g'); done)

flare_sensitivity_100seconds+eshift.json.gz:
	years=$$(printf "%.7f" $$(echo "100/3600/24/365" | bc -l)); \
    toise-figure-data pointsource.flare.sensitivity --emin=316. $$(for d in $(DETECTORS); do echo -d; for element in $$d; do echo $$element $$years; done; done) $$(for d in $(DETECTORS); do echo -o $(basename $(basename $@))_$$(echo $$d | sed -e 's| |+|g'); done)

flare_sensitivity_500seconds.json.gz:
	years=$$(printf "%.7f" $$(echo "500/3600/24/365" | bc -l)); \
    toise-figure-data pointsource.flare.sensitivity $$(for d in $(DETECTORS); do echo -d; for element in $$d; do echo $$element $$years; done; done) $$(for d in $(DETECTORS); do echo -o $(basename $(basename $@))_$$(echo $$d | sed -e 's| |+|g'); done)

flare_sensitivity_1000seconds.json.gz:
	years=$$(printf "%.7f" $$(echo "1000/3600/24/365" | bc -l)); \
    toise-figure-data pointsource.flare.sensitivity $$(for d in $(DETECTORS); do echo -d; for element in $$d; do echo $$element $$years; done; done) $$(for d in $(DETECTORS); do echo -o $(basename $(basename $@))_$$(echo $$d | sed -e 's| |+|g'); done)

flare_sensitivity_1day.json.gz:
	years=$$(printf "%.6f" $$(echo "1/365" | bc -l)); \
    toise-figure-data pointsource.flare.sensitivity $$(for d in $(DETECTORS); do echo -d; for element in $$d; do echo $$element $$years; done; done) $$(for d in $(DETECTORS); do echo -o $(basename $(basename $@))_$$(echo $$d | sed -e 's| |+|g'); done)

flare_sensitivity_14days.json.gz:
	years=$$(printf "%.5f" $$(echo "14/365" | bc -l)); \
    toise-figure-data pointsource.flare.sensitivity $$(for d in $(DETECTORS); do echo -d; for element in $$d; do echo $$element $$years; done; done) $$(for d in $(DETECTORS); do echo -o $(basename $(basename $@))_$$(echo $$d | sed -e 's| |+|g'); done)

flare_sensitivity_100days.json.gz:
	years=$$(printf "%.3f" $$(echo "100/365" | bc -l)); \
    toise-figure-data pointsource.flare.sensitivity $$(for d in $(DETECTORS); do echo -d; for element in $$d; do echo $$element $$years; done; done) $$(for d in $(DETECTORS); do echo -o $(basename $(basename $@))_$$(echo $$d | sed -e 's| |+|g'); done)
	
flare_sensitivity_10years.json.gz:
	years=$$(printf "%.3f" $$(echo "10" | bc -l)); \
    toise-figure-data pointsource.flare.sensitivity $$(for d in $(DETECTORS); do echo -d; for element in $$d; do echo $$element $$years; done; done) $$(for d in $(DETECTORS); do echo -o $(basename $(basename $@))_$$(echo $$d | sed -e 's| |+|g'); done)

flare_sensitivity_10years+eshift.json.gz:
	years=$$(printf "%.3f" $$(echo "10" | bc -l)); \
    toise-figure-data pointsource.flare.sensitivity --emin=316. $$(for d in $(DETECTORS); do echo -d; for element in $$d; do echo $$element $$years; done; done) $$(for d in $(DETECTORS); do echo -o $(basename $(basename $@))_$$(echo $$d | sed -e 's| |+|g'); done)

tdr: flare_sensitivity_100seconds.json.gz flare_sensitivity_100seconds+eshift.json.gz flare_sensitivity_500seconds.json.gz flare_sensitivity_1000seconds.json.gz flare_sensitivity_14days.json.gz flare_sensitivity_100days.json.gz flare_sensitivity_10years.json.gz flare_sensitivity_10years+eshift.json.gz

.PHONY: tdr
