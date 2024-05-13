
all: muon_performance.pdf surface_geometry.pdf cascade_aeff.pdf sparse_veto.pdf

muon_performance.pdf:
	toise-plot detector.tracks.performance -o $@

surface_geometry.pdf:
	toise-plot detector.surface_geometry -o $@


gen2_tdr_effective_areas.json.gz:
	toise-figure-data detector.effective_areas -d IceCube 0 Gen2-InIce 0 Gen2-Radio 0 -o $@

gen2_tdr_effective_areas.pdf: gen2_tdr_effective_areas.json.gz
	toise-plot detector.effective_areas $^ -o $@

sparse_veto.pdf :
	toise-plot detector.cascades.sparse_veto -o $@
