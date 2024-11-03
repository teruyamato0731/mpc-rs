extern crate nalgebra as na;

use mpc::ukf2::UnscentedKalmanFilter;
use na::{matrix, vector};
use rand_distr::{Distribution, Normal};
use std::{f64::consts::PI, thread::sleep};

const M1: f64 = 150e-3;
const R_W: f64 = 50e-3;
const M2: f64 = 2.3 - 2.0 * M1 + 2.0;
const L: f64 = 0.2474; // 重心までの距離
const J1: f64 = M1 * R_W * R_W;
const J2: f64 = 0.2;
const G: f64 = 9.81;
const KT: f64 = 0.15; // m2006

const DT: f64 = 0.01;
const Q: na::Matrix6<f64> = matrix![
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 0.0, 10.0;
];
const R: na::Matrix5<f64> = matrix![
    100.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 100.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.5, 0.0, 0.0;
    0.0, 0.0, 0.0, 100.0, 0.0;
    0.0, 0.0, 0.0, 0.0, 100.0;
];

// 系ダイナミクスを記述
fn fx(x: &na::Vector6<f64>, u: f64) -> na::Vector6<f64> {
    let mut r = *x;
    const D: f64 = (M1 + M2 + J1 / (R_W * R_W)) * (M2 * L * L + J2);
    let d = D - (M2 * L * x[2].cos()).powi(2);
    r[0] += x[1] * DT;
    r[1] += x[2] * DT;
    let term3 = (J2 + M2 * L * L) * (KT * u / R_W + M2 * L * x[4].powi(2) * x[3].sin());
    let term4 = M2 * G * L * L * x[3].sin() * x[3].cos();
    r[2] = (term3 + term4) / d;
    r[3] += x[4] * DT;
    r[4] += x[5] * DT;
    let term1 = (M1 + M2 + J1 / (R_W * R_W)) * M2 * G * L * x[3].sin();
    let term2 = (KT * u / R_W + M2 * L * x[4].powi(2) * x[3].sin()) * M2 * L * x[3].cos();
    r[5] = (term1 - term2) / d;
    r
}

// 観測関数
fn hx(x_act: &na::Vector6<f64>) -> na::Vector5<f64> {
    let v = M2 * G * x_act[3].cos() + M2 * x_act[2] * x_act[3].sin() - M2 * L * x_act[4].powi(2);
    let h = -M2 * G * x_act[3].sin() + M2 * x_act[2] * x_act[3].cos() + M2 * L * x_act[5];
    vector![
        60.0 / (2.0 * PI * R_W) * x_act[1], // 駆動輪のオドメトリ [m/s] -> [rpm]
        60.0 / (2.0 * PI * R_W) * x_act[1], // 駆動輪のオドメトリ [m/s] -> [rpm]
        x_act[3].to_degrees(),              // 角速度 [rad/s] -> [deg/s]
        v / G,                              // 垂直方向の力 [N] -> [G]
        h / G,                              // 水平方向の力 [N] -> [G]
    ]
}

// センサ出力をシミュレーション
fn sensor(x_act: na::Vector6<f64>, rng: &mut rand::rngs::ThreadRng) -> na::Vector5<f64> {
    let dist = Normal::new(0.0, 1.0).unwrap();
    let noise = vector![
        100.0 * dist.sample(rng),
        100.0 * dist.sample(rng),
        0.5 * dist.sample(rng),
        100.0 * dist.sample(rng),
        100.0 * dist.sample(rng),
    ];
    hx(&x_act) + noise
}

fn main() {
    let mut rng = rand::thread_rng();

    let mut x_act = vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let x_est = x_act;
    let p = matrix![
        10.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 10.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 10.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 10.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 10.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.0, 10.0;
    ];
    let mut ukf = UnscentedKalmanFilter::new(x_est, p, Q, R);
    for i in 0..100 {
        let u = vector![0.1];
        x_act = fx(&x_act, u[0]);
        ukf.predict(u[0], fx);
        let x_obs = sensor(x_act, &mut rng);
        ukf.update(&x_obs, hx);
        let x_est = ukf.state();
        let p = ukf.covariance();

        print!("t: {:4.2} ", i as f64 * DT);
        print!(
            "x_act: ({:7.2},{:7.2},{:7.2},{:7.2}) ",
            x_act[0], x_act[1], x_act[3], x_act[4]
        );
        print!(
            "x_obs: ({:7.2},{:7.2},{:7.2},{:7.2},{:7.2}) ",
            x_obs[0], x_obs[1], x_obs[2], x_obs[3], x_obs[4]
        );
        print!(
            "x_est: ({:7.2},{:7.2},{:7.2},{:7.2}) ",
            x_est[0], x_est[1], x_est[3], x_est[4]
        );
        println!(
            "p: ({:7.2},{:7.2},{:7.2},{:7.2},{:7.2},{:7.2})",
            p[(0, 0)],
            p[(1, 1)],
            p[(2, 2)],
            p[(3, 3)],
            p[(4, 4)],
            p[(5, 5)],
        );
    }

    // print完了を待つ
    sleep(std::time::Duration::from_millis(10));
}
