#!/bin/sh
# Reproduces the raw Horizons responses in this directory.
# usage: ./fetch.sh <NAIF id> <JD TDB> <output file>
# Frame: Saturn-centred (500@699), reference plane = Saturn body equator of date (IAU_SATURN), TDB.
BASE='https://ssd.jpl.nasa.gov/api/horizons.api?format=text'
Q="&COMMAND='$1'&OBJ_DATA='YES'&MAKE_EPHEM='YES'&EPHEM_TYPE='VECTORS'&CENTER='500@699'&REF_PLANE='BODY'&REF_SYSTEM='ICRF'&TLIST='$2'&TLIST_TYPE='JD'&TIME_TYPE='TDB'&VEC_TABLE='2'&OUT_UNITS='KM-S'&CSV_FORMAT='NO'&VEC_CORR='NONE'"
curl -sg -m 60 -o "$3" "$BASE$Q"
