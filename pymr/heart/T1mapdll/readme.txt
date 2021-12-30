cl /LD T1map2.c lmcurve.c lmmin.c
gcc  -g -shared T1map_linux.c lmcurve.c lmmin.c -o T1map.so -lm -fPIC
