
all: muon_performance.pdf surface_geometry.pdf cascade_aeff.pdf sparse_veto.pdf

muon_performance.pdf:
	toise-plot detector.tracks.performance -o $@

surface_geometry.pdf:
	toise-plot detector.surface_geometry -o $@

cascade_aeff.IceCube.json.gz:
	toise-figure-data detector.cascades.effective_area -d IceCube 0 -o $@

cascade_aeff.Gen2-InIce.json.gz:
	toise-figure-data detector.cascades.effective_area -d Gen2-InIce 0 -o $@

cascade_aeff.pdf : cascade_aeff.IceCube.json.gz cascade_aeff.Gen2-InIce.json.gz
	toise-plot detector.cascades.effective_area $^ -o $@

sparse_veto.pdf :
	toise-plot detector.cascades.sparse_veto -o $@
