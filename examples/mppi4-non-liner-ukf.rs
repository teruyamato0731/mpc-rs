extern crate nalgebra as na;
use mpc::mppi::Mppi;
use mpc::ukf2::UnscentedKalmanFilter;
use na::{matrix, vector};
use rand_distr::Distribution;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// MARK: - Constants
// 予測ホライゾン
const T: f64 = 1.2;
const N: usize = 8;
const DT: f64 = T / N as f64;

// 制御ホライゾン
const K: usize = 5e5 as usize;
const LAMBDA: f64 = 1.4;
const R_U: f64 = 4.0;
const C: na::Vector4<f64> = vector![0.1, 0.1, 1.0, 0.5];

// 制約
const LIMIT: (f64, f64) = (-10.0, 10.0);

// UKF
const PHY: na::Vector3<f64> = vector![100.0, 70.0, 20.0];
const R: na::SVector<f64, 5> = vector![200.0, 200.0, 10.0, 0.05, 0.05];

const DEBUG: bool = false;
const DEBUG_UKF: bool = true;

fn cost(x: &na::Vector4<f64>) -> f64 {
    C[0] * x[0].powi(2) + C[1] * x[1].powi(2) + C[2] * x[2].powi(2) + C[3] * x[3].powi(2)
}

// MARK: - Main
fn main() {
    let init_x = vector![0.0, 0.0, 0.0, 0.1, 0.0, 0.0];
    let init_u_n = na::SVector::<f64, N>::zeros();

    let mut mppi = Mppi::<N, K, 4>::new(|x, u| dynamics4(x, u, DT), cost, LAMBDA, R_U, LIMIT);

    let x_mutex = Arc::new(Mutex::new(init_x));
    let ukf_mutex = init_ukf(&init_x);
    let u_n_mutex = Arc::new(Mutex::new(init_u_n));

    start_dynamics_thread(x_mutex.clone(), u_n_mutex.clone());

    start_ukf_thread(x_mutex.clone(), u_n_mutex.clone(), ukf_mutex.clone());
    start_logging_thread(x_mutex.clone(), u_n_mutex.clone(), ukf_mutex.clone());

    let start = Instant::now();
    loop {
        let x_est = if DEBUG_UKF {
            let x = x_mutex.lock().unwrap();
            *x
        } else {
            let ukf = ukf_mutex.lock().unwrap();
            ukf.state()
        };

        // θの絶対値がpi/2を超えればエラー
        if x_est[3].abs() > std::f64::consts::PI / 2.0 {
            println!("θ is over pi/2");
            println!(
                "x: [{:6.2}, {:5.2}, {:5.2}, {:5.2}, {:5.2}, {:5.2}] ",
                x_est[0], x_est[1], x_est[2], x_est[3], x_est[4], x_est[5]
            );
            println!("elapsed: {:.2} sec", start.elapsed().as_secs_f64());
            break;
        }

        let pre_u_n = {
            let u_n = u_n_mutex.lock().unwrap();
            *u_n
        };

        let x_est = vector![x_est[0], x_est[1], x_est[3], x_est[4]];
        let mut u_n = match mppi.compute(&x_est, &pre_u_n) {
            Ok(u) => u,
            Err(e) => {
                println!("Failed to compute MPPI: {:?}", e);
                na::SVector::<f64, N>::zeros()
            }
        };

        if approx_equal(pre_u_n[0], u_n[0]) {
            continue;
        }

        if DEBUG {
            u_n[0] = 0.0;
        }

        {
            let mut tmp = u_n_mutex.lock().unwrap();
            *tmp = u_n;
        }

        print_con(start, &u_n, &x_est);
    }
}

// MARK: - Dynamics
// 系ダイナミクスを記述
/// 駆動輪の質量
const M1: f64 = 160e-3;
/// 駆動輪の半径
const R_W: f64 = 50e-3;
/// 振り子の質量
const M2: f64 = 2.4;
/// 振り子の長さ
const L: f64 = 0.4;
/// タイヤの慣性モーメント
const J1: f64 = 2.23e5 * 1e-9;
/// 振り子の慣性モーメント
const J2: f64 = 1.168e8 * 1e-9;
/// 重力加速度
const G: f64 = 9.81;
/// モータ定数
const KT: f64 = 0.15; // m2006 * 2
/// 分母係数
const D1: f64 = (2.0 * M1 + M2 + 2.0 * J1 / (R_W * R_W)) * (M2 * L * L + J2);
// 2階微分方程式を記述
fn ddot(x: &na::Vector4<f64>, u: f64) -> (f64, f64) {
    let d = D1 - (M2 * L * x[2].cos()).powi(2);
    let term1 = (M2 * L * L + J2) * M2 * L / d * x[3].powi(2) * x[2].sin();
    let term2 = -(M2 * L).powi(2) * G / d * x[2].sin() * x[2].cos();
    let term3 = 2.0 * (M2 * L * L + J2) / (d * R_W) * KT * u;
    let ddot_x = term1 + term2 + term3;
    let term1 = -(M2 * L).powi(2) / d * x[3].powi(2) * x[2].sin() * x[2].cos();
    let term2 = M2 * G * L * (2.0 * M1 + M2 + 2.0 * J1 / (R_W * R_W)) / d * x[2].sin();
    let term3 = -2.0 * M2 * L / (d * R_W) * KT * u * x[2].cos();
    let ddot_theta = term1 + term2 + term3;
    (ddot_x, ddot_theta)
}
fn dynamics4(x: &na::Vector4<f64>, u: f64, dt: f64) -> na::Vector4<f64> {
    let (ddot_x, ddot_theta) = ddot(x, u);
    let mut r = *x;
    r[3] += ddot_theta * dt;
    r[2] += r[3] * dt;
    r[1] += ddot_x * dt;
    r[0] += r[1] * dt;
    r
}
fn dynamics_short(x: &na::Vector6<f64>, u: f64, dt: f64) -> na::Vector6<f64> {
    let (ddot_x, ddot_theta) = ddot(&vector![x[0], x[1], x[3], x[4]], u);
    let mut r = *x;
    r[5] = ddot_theta;
    r[4] += r[5] * dt;
    r[3] += r[4] * dt;
    r[2] = ddot_x;
    r[1] += r[2] * dt;
    r[0] += r[1] * dt;
    r
}

// MARK: - UKF
fn init_ukf(init: &na::Vector6<f64>) -> Arc<Mutex<UnscentedKalmanFilter>> {
    let p = na::SMatrix::<f64, 6, 6>::identity() * 10.0;
    let r = na::SMatrix::<f64, 5, 5>::from_diagonal(&R);
    let q = gen_q(DT);
    let obj = UnscentedKalmanFilter::new(*init, p, q, r);
    Arc::new(Mutex::new(obj))
}
fn hx(state: &na::Vector6<f64>) -> na::Vector5<f64> {
    let ax = G * state[3].sin() + state[2] * state[3].cos() + L * state[5];
    let az = G * state[3].cos() - state[2] * state[3].sin() + L * state[4].powi(2);
    vector![
        36.0 * 60.0 / (2.0 * PI * R_W) * state[1], // 駆動輪のオドメトリ [m/s] -> [rpm]
        36.0 * -60.0 / (2.0 * PI * R_W) * state[1], // 駆動輪のオドメトリ [m/s] -> [rpm]
        state[4].to_degrees(),                     // 角速度 [rad/s] -> [deg/s]
        az / G,                                    // 垂直方向の加速度 [m/s^2] -> [G]
        ax / G,                                    // 水平方向の加速度 [m/s^2] -> [G]
    ]
}
fn sensor(x: &na::Vector6<f64>) -> na::Vector5<f64> {
    let mut rng = rand::thread_rng();
    let dist = rand_distr::Normal::<f64>::new(0.0, 1.0).unwrap();
    let noise = vector![
        R[0] * dist.sample(&mut rng),
        R[1] * dist.sample(&mut rng),
        R[2] * dist.sample(&mut rng),
        R[3] * dist.sample(&mut rng),
        R[4] * dist.sample(&mut rng),
    ];
    hx(x) + noise
}
fn gen_q(dt: f64) -> na::SMatrix<f64, 6, 6> {
    let dt_2 = dt.powi(2);
    let dt_3 = dt_2 * dt;
    let dt_4 = dt_2.powi(2);
    let q1 = matrix![
        0.0, 0.0, 0.0, 0.0       , 0.0       , 0.0       ;
        0.0, 0.0, 0.0, 0.0       , 0.0       , 0.0       ;
        0.0, 0.0, 0.0, 0.0       , 0.0       , 0.0       ;
        0.0, 0.0, 0.0, 0.0       , dt_4 / 8.0, dt_3 / 6.0;
        0.0, 0.0, 0.0, dt_4 / 8.0, dt_3 / 3.0, dt_2 / 2.0;
        0.0, 0.0, 0.0, dt_3 / 6.0, dt_2 / 2.0, dt        ;
    ];
    let q2 = matrix![
        0.0, 0.0       , 0.0, 0.0       , 0.0       , 0.0;
        0.0, 0.0       , 0.0, dt_4 / 8.0, dt_3 / 6.0, 0.0;
        0.0, 0.0       , 0.0, 0.0       , 0.0       , 0.0;
        0.0, dt_4 / 8.0, 0.0, dt_3 / 3.0, dt_2 / 2.0, 0.0;
        0.0, dt_3 / 6.0, 0.0, dt_2 / 2.0, dt        , 0.0;
        0.0, 0.0       , 0.0, 0.0       , 0.0       , 0.0;
    ];
    let q3 = matrix![
        0.0       , dt_4 / 8.0, dt_3 / 6.0, 0.0, 0.0, 0.0;
        dt_4 / 8.0, dt_3 / 3.0, dt_2 / 2.0, 0.0, 0.0, 0.0;
        dt_3 / 6.0, dt_2 / 2.0, dt        , 0.0, 0.0, 0.0;
        0.0       , 0.0       , 0.0       , 0.0, 0.0, 0.0;
        0.0       , 0.0       , 0.0       , 0.0, 0.0, 0.0;
        0.0       , 0.0       , 0.0       , 0.0, 0.0, 0.0;
    ];
    PHY[0] * q1 + PHY[1] * q2 + PHY[2] * q3
}

// MARK: - App
fn start_dynamics_thread(
    x_mutex: Arc<Mutex<na::Vector6<f64>>>,
    u_n_mutex: Arc<Mutex<na::SVector<f64, N>>>,
) {
    thread::spawn(move || {
        let start = Instant::now();
        let mut pre = start;
        loop {
            {
                let u = {
                    let u_n = u_n_mutex.lock().unwrap();
                    u_n[0]
                };
                let mut x = x_mutex.lock().unwrap();
                *x = dynamics_short(&x, u, pre.elapsed().as_secs_f64());
            }
            pre = Instant::now();
        }
    });
}

fn start_ukf_thread(
    x: Arc<Mutex<na::Vector6<f64>>>,
    u_n_mutex: Arc<Mutex<na::SVector<f64, N>>>,
    ukf_mutex: Arc<Mutex<UnscentedKalmanFilter>>,
) {
    thread::spawn(move || {
        // データが読み込まれるまで待機
        let start = Instant::now();
        let mut pre = start;
        loop {
            let x = {
                // ロックを取得できるまで待機
                let x = x.lock().expect("Failed to lock");
                *x
            };
            let x_obs = sensor(&x);
            // センサの遅延
            thread::sleep(Duration::from_millis(9));
            let u = {
                let u_n = u_n_mutex.lock().unwrap();
                u_n[0]
            };
            let (x_est, p) = {
                // ロックを取得できるまで待機
                let mut ukf = ukf_mutex.lock().expect("Failed to lock");
                let dt = pre.elapsed().as_secs_f64();
                pre = Instant::now();
                let fx = |x: &_, u: f64| dynamics_short(x, u, dt);
                let q = gen_q(dt);
                ukf.set_q(q);
                ukf.predict(u, fx);
                ukf.update(&x_obs, hx);
                (ukf.state(), ukf.covariance())
            };
            print_rcv(&x_est, &x_obs, start, u, &x, &p);
        }
    });
}

// MARK: - Print
fn print_con(start: Instant, u_n: &na::SVector<f64, N>, x_est: &na::SVector<f64, 4>) {
    print!("\x1b[32mCon:");
    print!("{:6.2} ", start.elapsed().as_secs_f64());
    print!("u:{:6.2} ", u_n[0]);
    print!(
        "e:[{:6.2},{:6.2},{:5.0},{:5.0}] ",
        x_est[0],
        x_est[1],
        x_est[2].to_degrees(),
        x_est[3].to_degrees()
    );
    println!("\x1b[m");
}
fn print_rcv(
    x_est: &na::SVector<f64, 6>,
    x_obs: &na::SVector<f64, 5>,
    start: Instant,
    u: f64,
    x: &na::SVector<f64, 6>,
    p: &na::SMatrix<f64, 6, 6>,
) {
    let h = hx(x_est);
    let z = x_obs - h;
    print!("\x1b[36mRcv:\x1b[m");
    print!("{:6.2} ", start.elapsed().as_secs_f64());
    print!("u:{:6.2} ", u);
    print!(
        "e:[{:6.2},{:6.2},{:5.0},{:5.0}] ",
        x_est[0],
        x_est[1],
        x_est[3].to_degrees(),
        x_est[4].to_degrees()
    );
    print!(
        "x:[{:6.2},{:6.2},{:5.0},{:5.0}] ",
        x[0],
        x[1],
        x[3].to_degrees(),
        x[4].to_degrees()
    );
    print!(
        "o:[{:6.0},{:6.0},{:4.0},{:5.2},{:5.2}] ",
        x_obs[0], x_obs[1], x_obs[2], x_obs[3], x_obs[4]
    );
    print!(
        "z:[{:6.0},{:6.0},{:4.0},{:5.2},{:5.2}] ",
        z[0], z[1], z[2], z[3], z[4]
    );
    print!(
        "p:[{:6.2},{:5.2},{:5.2},{:5.2},{:5.2},{:5.2}] ",
        p[(0, 0)],
        p[(1, 1)],
        p[(2, 2)],
        p[(3, 3)],
        p[(4, 4)],
        p[(5, 5)],
    );
    println!();
}

fn approx_equal(a: f64, b: f64) -> bool {
    let epsilon = 1e-2;
    (a - b).abs() < epsilon
}

// MARK: - Log
fn write(
    wtr: &mut csv::Writer<std::fs::File>,
    t: f64,
    u: f64,
    x: na::Vector6<f64>,
    x_est: na::Vector6<f64>,
    x_pred: na::Vector6<f64>,
) -> Result<(), csv::Error> {
    wtr.write_record(&[
        t.to_string(),
        u.to_string(),
        x[0].to_string(),
        x[1].to_string(),
        x[2].to_string(),
        x[3].to_string(),
        x[4].to_string(),
        x[5].to_string(),
        x_est[0].to_string(),
        x_est[1].to_string(),
        x_est[2].to_string(),
        x_est[3].to_string(),
        x_est[4].to_string(),
        x_est[5].to_string(),
        x_pred[0].to_string(),
        x_pred[1].to_string(),
        x_pred[2].to_string(),
        x_pred[3].to_string(),
        x_pred[4].to_string(),
        x_pred[5].to_string(),
    ])?;
    wtr.flush()?;
    Ok(())
}

fn start_logging_thread(
    x_mutex: Arc<Mutex<na::Vector6<f64>>>,
    u_n_mutex: Arc<Mutex<na::SVector<f64, N>>>,
    ukf_mutex: Arc<Mutex<UnscentedKalmanFilter>>,
) {
    thread::spawn(move || {
        let file_path = "logs/mppi/mppi.csv";
        let mut wtr = csv::Writer::from_path(file_path).expect("Failed to create file");
        let start = Instant::now();
        let mut pre_write = start;
        loop {
            // ログの書き込み 一定周期ごと
            if pre_write.elapsed() > Duration::from_millis(30) {
                pre_write = Instant::now();

                let u_n = {
                    let u_n = u_n_mutex.lock().unwrap();
                    u_n
                };
                let x = {
                    let x = x_mutex.lock().unwrap();
                    *x
                };
                let x_est = {
                    let ukf = ukf_mutex.lock().unwrap();
                    ukf.state()
                };

                let mut x_pred = x_est;
                for i in 0..N {
                    x_pred = dynamics_short(&x_pred, u_n[i], DT);
                }

                write(
                    &mut wtr,
                    start.elapsed().as_secs_f64(),
                    u_n[0],
                    x,
                    x_est,
                    x_pred,
                )
                .expect("Failed to write");
            }
        }
    });
}
