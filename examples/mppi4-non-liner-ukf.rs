extern crate nalgebra as na;
use mpc::mppi::Mppi;
use mpc::ukf2::UnscentedKalmanFilter;
use na::{matrix, vector};
use rand_distr::Distribution;
use std::f64::consts::PI;

// 予測ホライゾン
const T: f64 = 0.8;
const N: usize = 8;
const DT: f64 = T / N as f64;

// 制御ホライゾン
const K: usize = 15e5 as usize;
const LAMBDA: f64 = 0.5;
const R: f64 = 10.0;

// 制約
const LIMIT: (f64, f64) = (-10.0, 10.0);

fn cost(x: &na::Vector4<f64>) -> f64 {
    let x_clamped = x[0].clamp(-2.0, 2.0);
    let term1 = 2.0 * x_clamped.powi(2);
    let term2 = 3.0 * (x[1] + 2.0 * x_clamped).clamp(-5.0, 5.0).powi(2);
    let term3 = 5.0 * (x[2] + 0.35 * x[0].clamp(-0.75, 0.75)).powi(2);
    let term4 = 1.2 * x[3].powi(2);
    term1 + term2 + term3 + term4
}

fn main() {
    let mut x = vector![0.5, 0.0, 0.0, 0.1, 0.0, 0.0];
    let mut u_n = na::SVector::<f64, N>::zeros();

    let mut mppi = Mppi::<N, K>::new(fx, cost, LAMBDA, R, LIMIT);
    let mut ukf = init_ukf(&x);

    // ログファイルの作成
    let file_path = "logs/mppi/mppi.csv";
    let mut wtr = csv::Writer::from_path(file_path).expect("file open error");

    let now = std::time::Instant::now();
    let mut t = 0.0;
    while t < 10.0 {
        let x_est = ukf.state();
        let x_est = vector![x_est[0], x_est[1], x_est[2], x_est[3]];

        u_n = match mppi.compute(&x_est, &u_n) {
            Ok(u) => u,
            Err(e) => {
                println!("Failed to compute MPPI: {:?}", e);
                na::SVector::<f64, N>::zeros()
            }
        };

        ukf.predict(u_n[0], dynamics);
        x = dynamics(&x, u_n[0]);

        let x_obs = sensor(&x);
        ukf.update(&x_obs, hx);
        let x_est = ukf.state();

        print!("t: {:.2}, u: {:6.2} ", t, u_n[0]);
        print!(
            "x: [{:6.2}, {:5.2}, {:5.2}, {:5.2}] ",
            x[0], x[1], x[3], x[4]
        );
        print!(
            "ob: [{:6.0}, {:6.1}, {:6.1}, {:6.1}, {:6.1}] ",
            x_obs[0], x_obs[1], x_obs[2], x_obs[3], x_obs[4]
        );
        print!(
            "est: [{:6.2}, {:5.2}, {:5.2}, {:5.2}] ",
            x_est[0], x_est[1], x_est[3], x_est[4]
        );
        println!();

        // θ が 60度 以上になったら終了
        if x[3].abs() > 60.0f64.to_radians() {
            println!("x[2] is over 60 degrees");
            break;
        }

        wtr.write_record(&[
            t.to_string(),
            u_n[0].to_string(),
            x[0].to_string(),
            x[1].to_string(),
            x[3].to_string(),
            x[4].to_string(),
        ])
        .expect("write error");
        wtr.flush().expect("flush error");

        t += DT;
    }
    println!("elapsed: {:.2} sec", now.elapsed().as_secs_f64());
}

// 系ダイナミクスを記述
const M1: f64 = 150e-3;
const R_W: f64 = 50e-3;
const M2: f64 = 2.3 - 2.0 * M1 + 2.0;
const L: f64 = 0.2474; // 重心までの距離
const J1: f64 = M1 * R_W * R_W;
const J2: f64 = 0.2;
const G: f64 = 9.81;
const KT: f64 = 0.3; // m3508
const D: f64 = (M1 + M2 + J1 / (R_W * R_W)) * (M2 * L * L + J2);
fn dynamics(x: &na::Vector6<f64>, u: f64) -> na::Vector6<f64> {
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
fn fx(x: &na::Vector4<f64>, u: f64) -> na::Vector4<f64> {
    let mut r = *x;
    let d = D - M2 * M2 * L * L * x[2].cos() * x[2].cos();
    let term1 = (M1 + M2 + J1 / (R_W * R_W)) * M2 * G * L * x[2].sin();
    let term2 = (KT * u / R_W + M2 * L * x[3].powi(2) * x[2].sin()) * M2 * L * x[2].cos();
    r[3] += (term1 - term2) / d * DT;
    r[2] += x[3] * DT;
    let term3 = (J2 + M2 * L * L) * (KT * u / R_W + M2 * L * x[3].powi(2) * x[2].sin());
    let term4 = M2 * G * L * L * x[2].sin() * x[2].cos();
    r[1] += (term3 + term4) / d * DT;
    r[0] += x[1] * DT;
    r
}

fn init_ukf(init: &na::Vector6<f64>) -> UnscentedKalmanFilter {
    let p = matrix![
        3.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 3.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 3.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 3.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 3.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.0, 3.0;
    ];
    let q = matrix![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.0, 0.05;
        0.0, 0.0, 0.0, 0.0, 0.05, 5.0;
    ];
    let r = matrix![
        500.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 500.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 10.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 5.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 5.0;
    ];
    UnscentedKalmanFilter::new(*init, p, q, r)
}

fn sensor(x: &na::Vector6<f64>) -> na::Vector5<f64> {
    let mut rng = rand::thread_rng();
    let dist = rand_distr::Normal::<f64>::new(0.0, 1.0).unwrap();
    let noise = vector![
        100.0 * dist.sample(&mut rng),
        100.0 * dist.sample(&mut rng),
        0.5 * dist.sample(&mut rng),
        5.0 * dist.sample(&mut rng),
        5.0 * dist.sample(&mut rng),
    ];
    hx(x) + noise
}

fn hx(state: &na::Vector6<f64>) -> na::Vector5<f64> {
    let v = M2 * G * state[3].cos() + M2 * state[2] * state[3].sin() - M2 * L * state[4].powi(2);
    let h = -M2 * G * state[3].sin() + M2 * state[2] * state[3].cos() + M2 * L * state[5];
    vector![
        60.0 / (2.0 * PI * R_W) * state[1], // 駆動輪のオドメトリ [m/s] -> [rpm]
        60.0 / (2.0 * PI * R_W) * state[1], // 駆動輪のオドメトリ [m/s] -> [rpm]
        state[3].to_degrees(),              // 角速度 [rad/s] -> [deg/s]
        v / G,                              // 垂直方向の力 [N] -> [G]
        h / G,                              // 水平方向の力 [N] -> [G]
    ]
}
