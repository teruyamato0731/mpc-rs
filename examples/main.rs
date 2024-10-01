use std::thread::sleep;

use mpc::{Dynamics, Mpc};

fn main() {
    let model = Dynamics::default();
    let mut mpc = Mpc::new(model);

    loop {
        let force = mpc.update(1.0);
        println!("{:?}\t{:?}", force, mpc.dynamics);
        sleep(std::time::Duration::from_secs(1));
    }
}
