#!/bin/bash

cargo run -r --example mpc-ukf-commu | tee logs/mpc-ukf-commu/mpc-ukf-commu-$(date +%Y%m%d%H%M%S).log
