#!/bin/bash

cargo run -r --example mpc-ukf-commu | tee logs/mpc-ukf-com/mpc-ukf-com-$(date +%Y%m%d%H%M%S).log
# cargo run -r --example mppi4-ukf-commu | tee logs/mppi-ukf-com/mppi-ukf-com-$(date +%Y%m%d%H%M%S).log
