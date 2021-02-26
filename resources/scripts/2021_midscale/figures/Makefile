
.PHONY:: all

PHASE2_GEOMETRY := corner

disc_potential_$(PHASE2_GEOMETRY)_E-2.0.png: sensitivity_Sunflower_$(PHASE2_GEOMETRY)_-2.0.json gen2_ic86_1yr_sens_-2.0.json.gz
	python ../plot_sens.py -n $(PHASE2_GEOMETRY) -g -2.0

sensitivity_Sunflower_$(PHASE2_GEOMETRY)_-2.0.json:
	python ../calc_sens.py -n $(PHASE2_GEOMETRY) -g -2.0

gen2_ic86_1yr_sens_-2.0.json.gz:
	gen2-figure-data pointsource.flare.sensitivity -d Gen2-InIce-TracksOnly 1 IceCube-TracksOnly 1 --gamma -2 -o gen2_ic86_1yr_sens_-2.0

# ####################################
# Event counts above energy threshold
# #####################################

diffuse.flavor.event_counts.IceCube.json.gz:
	gen2-figure-data diffuse.flavor.event_counts -d IceCube 1 -o diffuse.flavor.event_counts.IceCube --astro 1 --gamma -2.0

diffuse.flavor.event_counts.Gen2.json.gz:
	gen2-figure-data diffuse.flavor.event_counts -d Gen2-InIce 1 -o diffuse.flavor.event_counts.Gen2 --astro 1 --gamma -2.0

diffuse.flavor.event_counts.Gen2Phase2-$(PHASE2_GEOMETRY).json.gz:
	gen2-figure-data diffuse.flavor.event_counts -d Gen2-Phase2-${PHASE2_GEOMETRY} 1 -o diffuse.flavor.event_counts.Gen2Phase2 --astro 1 --gamma -2.0

event_counts: diffuse.flavor.event_counts.IceCube.json.gz diffuse.flavor.event_counts.Gen2.json.gz diffuse.flavor.event_counts.Gen2Phase2-$(PHASE2_GEOMETRY).json.gz

# ####################################
# Flare sensitivity
# #####################################

# 100s, 1000s, 100d, 10y
DURATIONS=100 1000 8640000 315576000
define make-target

years_$1s := $(shell echo 'scale=9; $1/(365*86400)' | bc)
flare_sensitivity.$1s.json.gz:
	gen2-figure-data pointsource.flare.sensitivity -d IceCube $${years_$1s} Gen2-InIce $${years_$1s} Gen2-Phase2-$$(PHASE2_GEOMETRY) $${years_$1s} -o $$@

flare_sensitivity:: flare_sensitivity.$1s.json.gz

endef

$(foreach element,$(DURATIONS),$(eval $(call make-target,$(element))))

all:: flare_sensitivity
.PHONY:: flare_sensitivity

# ####################################
# Flare significance
# #####################################

flare_significance_energyrange.pdf: flare_significance_icecube_10yr_energyrange.json.gz flare_significance_gen2_10yr_energyrange.json.gz flare_significance_gen2phase2-$(PHASE2_GEOMETRY)_10yr_energyrange.json.gz
	gen2-plot pointsource.flare.single_flare_time_to_signficance $^ -o $@

flare_significance_energyrange.json: flare_significance_icecube_10yr_energyrange.json.gz flare_significance_gen2_10yr_energyrange.json.gz flare_significance_gen2phase2-$(PHASE2_GEOMETRY)_10yr_energyrange.json.gz
	gen2-plot pointsource.flare.single_flare_time_to_signficance $^ -d flare_significance_energyrange.json

flare_significance_icecube_10yr_energyrange.json.gz:
	gen2-figure-data pointsource.flare.single_flare_time_to_signficance -d IceCube-TracksOnly 10 --emin 3e4 --emax 3e6 -o flare_significance_icecube_10yr_energyrange

flare_significance_gen2_10yr_energyrange.json.gz:
	gen2-figure-data pointsource.flare.single_flare_time_to_signficance -d Gen2-InIce-TracksOnly 10 --emin 3e4 --emax 3e6 -o flare_significance_gen2_10yr_energyrange

flare_significance_gen2phase2-$(PHASE2_GEOMETRY)_10yr_energyrange.json.gz:
	gen2-figure-data pointsource.flare.single_flare_time_to_signficance -d Gen2-Phase2-$(PHASE2_GEOMETRY) 10 --emin 3e4 --emax 3e6 -o flare_significance_gen2phase2_10yr_energyrange

flare_significance: flare_significance_energyrange.json flare_significance_energyrange.pdf
all:: flare_significance
.PHONY:: flare_significance

# ####################################
# Flavor triangle contours
# #####################################

diffuse.flavor.confidence_levels.icecube_8yr.json.gz:
	gen2-figure-data diffuse.flavor.confidence_levels -d IceCube 8 -o $@

diffuse.flavor.confidence_levels.icecube_15yr.json.gz:
	gen2-figure-data diffuse.flavor.confidence_levels -d IceCube 15 -o $@

diffuse.flavor.confidence_levels.gen2phase2-$(PHASE2_GEOMETRY)_10yr.json.gz:
	gen2-figure-data diffuse.flavor.confidence_levels -d IceCube 15 Gen2-Phase2-$(PHASE2_GEOMETRY) 10 -o $@

flavor_triangle: diffuse.flavor.confidence_levels.icecube_8yr.json.gz diffuse.flavor.confidence_levels.icecube_15yr.json.gz diffuse.flavor.confidence_levels.gen2phase2_10yr.json.gz

.PHONY:: flavor_triangle
all:: flavor_triangle

# ####################################
# Spectrum unfolding
# #####################################

diffuse.spectrum.unfolding.10yr.json.gz:
	gen2-figure-data diffuse.spectrum.unfold --gzk-norm 0.1 -d IceCube 15 Gen2-Phase2-$(PHASE2_GEOMETRY) 10 Gen2-Phase2-Radio 10 -o $@ --clean

diffuse.spectrum.unfolding.10yr.pdf: diffuse.spectrum.unfolding.10yr.json.gz
	gen2-plot diffuse.spectrum.unfolded_flux --label 'Gen2-Phase2 InIce+Radio' $^ -o $@

unfolding: diffuse.spectrum.unfolding.10yr.pdf
.PHONY:: unfolding
all:: unfolding
