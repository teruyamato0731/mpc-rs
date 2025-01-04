extern crate nalgebra as na;
use mpc::mppi::Mppi;

// ref: https://zenn.dev/teruyamato0731/scraps/bc2b2b7c96bd07
// cargo run --example mppi4 --release

// 予測ホライゾン
const T: f64 = 0.8;
const N: usize = 8;
const DT: f64 = T / N as f64;

// 制御ホライゾン
const K: usize = 8e5 as usize;
const LAMBDA: f64 = 0.5;
const R: f64 = 3.0;

// 制約
const LIMIT: (f64, f64) = (-20.0, 20.0);

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

    let mut mppi = Mppi::<N, K, 4>::new(dynamics, cost, LAMBDA, R, LIMIT);

    // ログファイルの作成
    let file_path = "logs/mppi/mppi.csv";
    let mut wtr = csv::Writer::from_path(file_path).expect("file open error");

    let now = std::time::Instant::now();
    let mut t = 0.0;
    while t < 10.0 {
        u_n = mppi.compute(&x, &u_n).unwrap();
        x = dynamics(&x, u_n[0]);

        println!(
            "t: {:.2}, u: {:6.2}, x: [{:6.2}, {:5.2}, {:5.2}, {:5.2}]",
            t, u_n[0], x[0], x[1], x[2], x[3]
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
fn dynamics(x: &na::Vector4<f64>, u: f64) -> na::Vector4<f64> {
    let mut r = *x;
    const D: f64 = (M1 + M2 + J1 / (R_W * R_W)) * (M2 * L * L + J2);
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
