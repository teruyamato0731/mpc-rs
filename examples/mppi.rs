extern crate nalgebra as na;

use anyhow::{Ok, Result};
use na::vector;
use rand_distr::{Distribution, Normal};

// MPPI
// ref: https://zenn.dev/takuya_fukatsu/articles/36c0d6911d18b7

const T: f64 = 1.5;
const N: usize = 100;
const DT: f64 = T / N as f64;
const K: usize = 100;

type State = na::Vector2<f64>;
type StateSeq = na::SMatrix<f64, 2, N>;
type Control = na::Vector1<f64>;
type ControlSeq = na::SMatrix<f64, 1, N>;
type ControlSample = na::SMatrix<f64, N, K>;

const LIMIT: (f64, f64) = (-10.0, 10.0);

fn cost(x: &StateSeq, u: &ControlSeq) -> f64 {
    let mut c = 0.0;
    for i in 0..N {
        c += x[(0, i)].powi(2) + u[(0, i)].powi(2);
    }
    c
}

fn mppi(x: &State, u: &mut ControlSeq) -> Result<State> {
    let mut noise = ControlSample::new_random();
    // let mut u_seq = ControlSample::zeros();
    // u_seq += noise;
    for row in noise.row_iter_mut() {
        for e in row.iter_mut() {
            *e = e.clamp(LIMIT.0, LIMIT.1);
        }
    }
    let x_pred = predict(*x, *u);
    let c = cost(&x_pred, &u);
    // TODO: update u
    Ok(*x)
}

fn predict(mut x: State, u: ControlSeq) -> StateSeq {
    let mut x_seq = StateSeq::zeros();
    for i in 0..N {
        x = dynamics(x, u.column(i).into_owned());
        x_seq.column_mut(i).copy_from(&x);
    }
    x_seq
}

fn dynamics(mut x: State, u: Control) -> State {
    x[1] += u[0] * DT;
    x[0] += x[1] * DT;
    x
}

fn main() -> Result<()> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 2.0).unwrap();
    let _ = normal.sample(&mut rng);

    let mut u = ControlSeq::zeros();
    let mut x_act = vector![0.0, 0.0];

    for _ in 0..1000 {
        let x_pred = mppi(&x_act, &mut u)?;
        x_act = dynamics(x_act, u.column(0).into_owned());

        print!("{:?}\t{:?}", x_pred, x_act);
        println!();
    }

    Ok(())
}
