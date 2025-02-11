#!/bin/bash

# Interval in seconds (e.g., 3600 seconds = 1 hour)
INTERVAL=3600

while true; do
    # # Refresh Kerberos ticket
    # echo "Refreshing Kerberos ticket..."
    # kinit abhat@FNAL.GOV

    # Refresh kx509
    echo "Refreshing kx509..."
    /cvmfs/fermilab.opensciencegrid.org/products/common/prd/kx509/v3_1_1/NULL/bin/kx509

    # Refresh VOMS proxy
    echo "Refreshing VOMS proxy..."
    voms-proxy-init -noregen -rfc -voms 'fermilab:/fermilab/sbnd/Role=Analysis'

    # Wait for the specified interval
    echo "Waiting for $INTERVAL seconds before the next refresh..."
    sleep $INTERVAL
done
