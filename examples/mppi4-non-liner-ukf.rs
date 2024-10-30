extern crate nalgebra as na;
use std::f64::consts::PI;

use mpc::mppi::Mppi;
use mpc::ukf::UnscentedKalmanFilter;
use na::matrix;
use rand_distr::Distribution;

// 予測ホライゾン
const T: f64 = 0.8;
const N: usize = 8;
const DT: f64 = T / N as f64;

// 制御ホライゾン
const K: usize = 5e5 as usize;
const LAMBDA: f64 = 0.5;
const R: f64 = 3.0;

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
    let mut x = na::Vector4::new(0.5, 0.0, 0.1, 0.0);
    let mut u_n = na::SVector::<f64, N>::zeros();

    let mut mppi = Mppi::<N, K>::new(dynamics, cost, LAMBDA, R, LIMIT);
    let mut ukf = init_ukf(&x);

    // ログファイルの作成
    let file_path = "logs/mppi.csv";
    let mut wtr = csv::Writer::from_path(file_path).expect("file open error");

    let now = std::time::Instant::now();
    let mut t = 0.0;
    while t < 10.0 {
        let x_est = ukf.state();

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

        println!(
            "t: {:.2}, u: {:6.2}, x: [{:6.2}, {:5.2}, {:5.2}, {:5.2}], ob: [{:6.0}, {:6.1}], est: [{:6.2}, {:5.2}, {:5.2}, {:5.2}]",
            t, u_n[0], x[0], x[1], x[2], x[3], x_obs[0], x_obs[1], x_est[0], x_est[1], x_est[2], x_est[3]
        );

        // x[2] が 60度 以上になったら終了
        if x[2].abs() > 60.0f64.to_radians() {
            println!("x[2] is over 60 degrees");
            break;
        }

        wtr.write_record(&[
            t.to_string(),
            u_n[0].to_string(),
            x[0].to_string(),
            x[1].to_string(),
            x[2].to_string(),
            x[3].to_string(),
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
const KT: f64 = 0.15; // m2006
const D: f64 = (M1 + M2 + J1 / R_W * R_W) * (M2 * L * L + J2);
fn dynamics(x: &na::Vector4<f64>, u: f64) -> na::Vector4<f64> {
    let mut r = *x;
    let d = D - M2 * M2 * L * L * x[2].cos() * x[2].cos();
    let term1 = (M1 + M2 + J1 / R_W * R_W) * M2 * G * L * x[2].sin();
    let term2 = (KT * u / R_W + M2 * L * x[3].powi(2) * x[2].sin()) * M2 * L * x[2].cos();
    r[3] += (term1 - term2) / d * DT;
    r[2] += x[3] * DT;
    let term3 = (J2 + M2 * L * L) * (KT * u / R_W + M2 * L * x[3].powi(2) * x[2].sin());
    let term4 = M2 * G * L * L * x[2].sin() * x[2].cos();
    r[1] += (term3 + term4) / d * DT;
    r[0] += x[1] * DT;
    r
}

fn init_ukf(init: &na::Vector4<f64>) -> UnscentedKalmanFilter {
    let p = matrix![
        3.0, 0.0, 0.0, 0.0;
        0.0, 3.0, 0.0, 0.0;
        0.0, 0.0, 3.0, 0.0;
        0.0, 0.0, 0.0, 3.0;
    ];
    let q = matrix![
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.05;
        0.0, 0.0, 0.05, 5.0;
    ];
    let r = matrix![
        200.0, 0.0;
        0.0, 10.0;
    ];
    UnscentedKalmanFilter::new(*init, p, q, r)
}

fn sensor(x: &na::Vector4<f64>) -> na::Vector2<f64> {
    let mut rng = rand::thread_rng();
    let dist = rand_distr::Normal::<f64>::new(0.0, 1.0).unwrap();
    let noise = na::Vector2::new(100.0 * dist.sample(&mut rng), 0.5 * dist.sample(&mut rng));
    hx(x) + noise
}

fn hx(state: &na::Vector4<f64>) -> na::Vector2<f64> {
    na::Vector2::new(
        60.0 / (2.0 * PI * R_W) * state[1], // 駆動輪のオドメトリ [m/s] -> [rpm]
        state[3].to_degrees(),              // 角速度 [rad/s] -> [deg/s]
    )
}
