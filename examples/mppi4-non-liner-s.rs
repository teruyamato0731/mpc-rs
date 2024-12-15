extern crate nalgebra as na;
use mpc::mppi::Mppi;
use mpc::ukf::UnscentedKalmanFilter;
use na::matrix;
use rand_distr::Distribution;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

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
    let init = na::Vector4::new(0.0, 0.0, 0.01, 0.0);
    let x = Arc::new(Mutex::new(init));
    let x1 = x.clone();
    let init_u_n = na::SVector::<f64, N>::zeros();
    let u_n_mutex = Arc::new(Mutex::new(init_u_n));
    let u_n_mutex2 = u_n_mutex.clone();
    let u_n_mutex3 = u_n_mutex.clone();

    let mut mppi = Mppi::<N, K>::new(dynamics, cost, LAMBDA, R, LIMIT);
    let ukf_mutex = init_ukf(&init);
    let ukf_mutex2 = ukf_mutex.clone();

    // ログファイルの作成
    let file_path = "logs/mppi/mppi.csv";
    let mut wtr = csv::Writer::from_path(file_path).expect("file open error");

    thread::spawn(move || {
        let start = std::time::Instant::now();
        let mut pre = start;
        loop {
            {
                let mut x = x.lock().unwrap();
                let u_n = *u_n_mutex2.lock().unwrap();
                *x = dynamics_short(&x, u_n[0], pre.elapsed().as_secs_f64());
            }
            pre = std::time::Instant::now();
            thread::sleep(Duration::from_millis(1));
        }
    });

    thread::spawn(move || {
        // データが読み込まれるまで待機
        let start = std::time::Instant::now();
        let mut pre = start;
        loop {
            let x = {
                // ロックを取得できるまで待機
                let x = x1.lock().expect("Failed to lock");
                *x
            };
            let x_obs = sensor(&x);
            // センサの遅延
            thread::sleep(Duration::from_millis(1));
            let u_n = {
                let u_n = u_n_mutex3.lock().unwrap();
                *u_n
            };
            let (x_est, p) = {
                // ロックを取得できるまで待機
                let mut ukf = ukf_mutex.lock().expect("Failed to lock");
                let dt = pre.elapsed().as_secs_f64();
                pre = std::time::Instant::now();
                let fx = |x: &na::Vector4<f64>, u: f64| dynamics_short(x, u, dt);
                ukf.predict(u_n[0], fx);
                ukf.update(&x_obs, hx);
                (ukf.state(), ukf.covariance())
            };
            print!("\x1b[36mRcv: \x1b[m");
            print!("t: {:.2} ", start.elapsed().as_secs_f64());
            print!(
                "x: [{:6.2}, {:5.2}, {:5.2}, {:5.2}] ",
                x[0], x[1], x[2], x[3]
            );
            print!(
                "est: [{:6.2}, {:5.2}, {:5.2}, {:5.2}] ",
                x_est[0], x_est[1], x_est[2], x_est[3]
            );
            print!(
                "p: [{:6.2}, {:5.2}, {:5.2}, {:5.2}] ",
                p[(0, 0)],
                p[(1, 1)],
                p[(2, 2)],
                p[(3, 3)]
            );
            println!();
            // 次の送信まで待機
            thread::sleep(Duration::from_millis(2));
        }
    });

    let start = std::time::Instant::now();
    loop {
        let x_est = {
            // ロックを取得できるまで待機
            let ukf = ukf_mutex2.lock().expect("Failed to lock");
            ukf.state()
        };

        // x[2] が 60度 以上になったら終了
        if x_est[2].abs() > 60.0f64.to_radians() {
            println!("x[2] is over 60 degrees");
            break;
        }

        let u_n = {
            let u_n = u_n_mutex.lock().unwrap();
            *u_n
        };
        let result = mppi.compute(&x_est, &u_n);
        let u_n = match result {
            Ok(u) => u,
            _ => na::SVector::<f64, N>::zeros(),
        };
        {
            let mut tmp = u_n_mutex.lock().unwrap();
            *tmp = u_n;
        }

        print!("\x1b[32mCon: \x1b[m");
        print!("t: {:.2} ", start.elapsed().as_secs_f64());
        print!(
            "est: [{:6.2}, {:5.2}, {:5.2}, {:5.2}] ",
            x_est[0], x_est[1], x_est[2], x_est[3]
        );
        print!("u: {:8.3} ", u_n[0]);

        if let Err(e) = result {
            print!("\x1b[31mFailed to compute MPPI: {:?}\x1b[m", e);
        }

        println!();

        wtr.write_record(&[
            start.elapsed().as_secs_f64().to_string(),
            u_n[0].to_string(),
            x_est[0].to_string(),
            x_est[1].to_string(),
            x_est[2].to_string(),
            x_est[3].to_string(),
        ])
        .expect("write error");
        wtr.flush().expect("flush error");
    }
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

// 系ダイナミクスを記述
fn dynamics_short(x: &na::Vector4<f64>, u: f64, dt: f64) -> na::Vector4<f64> {
    let mut r = *x;
    const D: f64 = (M1 + M2 + J1 / (R_W * R_W)) * (M2 * L * L + J2);
    let d = D - M2 * M2 * L * L * x[2].cos() * x[2].cos();
    let term1 = (M1 + M2 + J1 / (R_W * R_W)) * M2 * G * L * x[2].sin();
    let term2 = (KT * u / R_W + M2 * L * x[3].powi(2) * x[2].sin()) * M2 * L * x[2].cos();
    r[3] += (term1 - term2) / d * dt;
    r[2] += x[3] * dt;
    let term3 = (J2 + M2 * L * L) * (KT * u / R_W + M2 * L * x[3].powi(2) * x[2].sin());
    let term4 = M2 * G * L * L * x[2].sin() * x[2].cos();
    r[1] += (term3 + term4) / d * dt;
    r[0] += x[1] * dt;
    r
}

fn init_ukf(init: &na::Vector4<f64>) -> Arc<Mutex<UnscentedKalmanFilter>> {
    let p = matrix![
        1.0, 0.0, 0.0, 0.0;
        0.0, 1.0, 0.0, 0.0;
        0.0, 0.0, 1.0, 0.0;
        0.0, 0.0, 0.0, 1.0;
    ];
    let q = matrix![
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 1.0;
        0.0, 0.0, 1.0, 1e2;
        0.0, 1.0, 1e2, 1e4;
    ];
    let r = matrix![
        50.0, 0.0, 0.0;
        0.0, 50.0, 0.0;
        0.0, 0.0, 0.5;
    ];
    let obj = UnscentedKalmanFilter::new(*init, p, q, r);
    Arc::new(Mutex::new(obj))
}

fn sensor(x: &na::Vector4<f64>) -> na::Vector3<f64> {
    let mut rng = rand::thread_rng();
    let dist = rand_distr::Normal::<f64>::new(0.0, 1.0).unwrap();
    let noise = na::Vector3::new(
        50.0 * dist.sample(&mut rng),
        50.0 * dist.sample(&mut rng),
        0.5 * dist.sample(&mut rng),
    );
    hx(x) + noise
}

fn hx(state: &na::Vector4<f64>) -> na::Vector3<f64> {
    na::Vector3::new(
        60.0 / (2.0 * PI * R_W) * state[1], // 駆動輪のオドメトリ [m/s] -> [rpm]
        60.0 / (2.0 * PI * R_W) * state[1], // 駆動輪のオドメトリ [m/s] -> [rpm]
        state[3].to_degrees(),              // 角速度 [rad/s] -> [deg/s]
    )
}
