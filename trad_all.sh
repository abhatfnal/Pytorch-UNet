#!/bin/bash

LD_PRELOAD=/lib64/libXrdPosixPreload.so:${LD_PRELOAD} python eval_trad.py --config config-nomMC_U.json --output traditional_nomMC_Plane0 --range 0 1600
LD_PRELOAD=/lib64/libXrdPosixPreload.so:${LD_PRELOAD} python eval_trad.py --config config-nomMC_V.json --output traditional_nomMC_Plane1 --range 0 1600
LD_PRELOAD=/lib64/libXrdPosixPreload.so:${LD_PRELOAD} python eval_trad.py --config config-randMC_U.json --output traditional_randMC_Plane0 --range 0 1600
LD_PRELOAD=/lib64/libXrdPosixPreload.so:${LD_PRELOAD} python eval_trad.py --config config-randMC_V.json --output traditional_randMC_Plane1 --range 0 1600
LD_PRELOAD=/lib64/libXrdPosixPreload.so:${LD_PRELOAD} python eval_trad.py --config config-opaqueMC_U.json --output traditional_opaqueMC_Plane0 --range 0 1600
LD_PRELOAD=/lib64/libXrdPosixPreload.so:${LD_PRELOAD} python eval_trad.py --config config-opaqueMC_V.json --output traditional_opaqueMC_Plane1 --range 0 1600