python calc_radio_sens.py -t 1.0 && mv sensitivity_radio_config_-2.0.json sensitivity_radio_config_1year_-2.0.json
python calc_radio_sens.py -t 10.0 && mv sensitivity_radio_config_-2.0.json sensitivity_radio_config_10years_-2.0.json
python calc_radio_sens.py -t 0.273973 && mv sensitivity_radio_config_-2.0.json sensitivity_radio_config_100days_-2.0.json
python calc_radio_sens.py -t 3.171e-6 && mv sensitivity_radio_config_-2.0.json sensitivity_radio_config_100s_-2.0.json
python calc_radio_sens.py -t 3.171e-5 && mv sensitivity_radio_config_-2.0.json sensitivity_radio_config_1000s_-2.0.json
