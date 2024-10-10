extern crate nalgebra as na;

use mpc::ukf::UnscentedKalmanFilter;
use na::{matrix, vector};
use rand_distr::{Distribution, Normal};

const M1: f64 = 150e-3;
const R_W: f64 = 50e-3;
const M2: f64 = 2.3 - 2.0 * M1 + 2.0;
const L: f64 = 0.2474; // 重心までの距離
const J1: f64 = M1 * R_W * R_W;
const J2: f64 = 0.1;
const G: f64 = 9.81;
const KT: f64 = 0.15; // m2006
const D: f64 = (M1 + M2 + J1 / R_W * R_W) * (M2 * L * L + J2) - M2 * M2 * L * L;

const DT: f64 = 0.01;
const Q: na::Matrix4<f64> = matrix![
    0.0, 0.0, 0.0, 0.0;
    0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 0.25, 0.5;
    0.0, 0.0, 0.5, 1.0;
];
const R: na::Matrix2<f64> = matrix![
    0.5, 0.0;
    0.0, 0.5;
];

// 状態遷移関数
fn fx(mut x: na::Vector4<f64>, u: f64) -> na::Vector4<f64> {
    x[3] += ((M1 + M2 + J1 / R_W * R_W) / D * M2 * G * L * x[2] - M2 * L / D / R_W * KT * u) * DT;
    x[2] += x[3] * DT;
    x[1] += (-M2 * M2 * G * L * L / D * x[2] + (M2 * L * L + J2) / D / R_W * KT * u) * DT;
    x[0] += x[1] * DT;
    x
}

// 観測関数
fn hx(x_act: na::Vector4<f64>) -> na::Vector2<f64> {
    vector![
        x_act[1], // 駆動輪のオドメトリ
        x_act[3], // 角速度
    ]
}

// センサ出力をシミュレーション
fn sensor(x_act: na::Vector4<f64>, rng: &mut rand::rngs::ThreadRng) -> na::Vector2<f64> {
    let mut x_obs = vector![
        x_act[1], // 駆動輪のオドメトリ
        x_act[3], // 角速度
    ];
    for i in 0..2 {
        let normal = Normal::new(0.0, R[(i, i)]).unwrap();
        x_obs[i] += normal.sample(rng);
    }
    x_obs
}

fn main() {
    let mut rng = rand::thread_rng();

    let mut x_act = vector![0.0, 0.0, 0.0, 0.0];
    let x_est = vector![0.0, 0.0, 0.0, 0.0];
    let p = matrix![
        10.0, 0.0, 0.0, 0.0;
        0.0, 10.0, 0.0, 0.0;
        0.0, 0.0, 10.0, 0.0;
        0.0, 0.0, 0.0, 10.0;
    ];
    let mut ukf = UnscentedKalmanFilter::new(x_est, p, Q, R);
    for _ in 0..100 {
        let u = vector![0.0015];
        x_act = fx(x_act, u[0]);
        ukf.predict(u[0], fx);
        let x_obs = sensor(x_act, &mut rng);
        ukf.update(&x_obs, hx);
        let x_est = ukf.state();

        print!(
            "x_act: ({:7.2},{:7.2},{:7.2},{:7.2}) ",
            x_act[0], x_act[1], x_act[2], x_act[3]
        );
        print!("x_obs: ({:7.2},{:7.2}) ", x_obs[0], x_obs[1]);
        print!(
            "x_est: ({:7.2},{:7.2},{:7.2},{:7.2}) ",
            x_est[0], x_est[1], x_est[2], x_est[3]
        );
        println!(
            "p: ({:7.2},{:7.2},{:7.2},{:7.2})",
            p[(0, 0)],
            p[(1, 1)],
            p[(2, 2)],
            p[(3, 3)]
        );
    }
}
