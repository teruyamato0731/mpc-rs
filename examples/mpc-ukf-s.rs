extern crate nalgebra as na;
use mpc::ukf2::UnscentedKalmanFilter;
use mpc::{create_f_matrix, create_g_matrix, create_q_matrix};
use na::{matrix, vector};
use optimization_engine::{panoc::*, *};
use rand_distr::Distribution as _;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// 予測ホライゾン
const T: f64 = 1.2;
const N: usize = 40;
const DT: f64 = T / N as f64;
const DUR: f64 = 0.03;

// 制約
const LIMIT: (f64, f64) = (-10.0, 10.0);
const C: na::Matrix4<f64> = matrix![
    1.0, 0.0, 0.0, 0.0;
    0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 10.0, 0.0;
    0.0, 0.0, 0.0, 5.0
];

// UKF
const PHY: f64 = 0.5e6;
const R: na::SVector<f64, 5> = vector![1500.0, 1500.0, 10.0, 0.5, 0.5];

// MARK: - Main
fn main() {
    let tolerance = 1e-6;
    let lbfgs_memory = 20;
    let mut panoc_cache = PANOCCache::new(N, tolerance, lbfgs_memory);

    let init_u_n = na::SVector::<f64, N>::zeros();
    let u_n_mutex = Arc::new(Mutex::new(init_u_n));
    let init_x = vector![0.5, 0.0, 0.0, 0.1, 0.0, 0.0];
    let x_mutex = Arc::new(Mutex::new(init_x));
    let ukf_mutex = init_ukf(&init_x);

    start_dynamics_thread(x_mutex.clone(), u_n_mutex.clone());

    start_ukf_thread(x_mutex.clone(), u_n_mutex.clone(), ukf_mutex.clone());
    start_logging_thread(x_mutex.clone(), u_n_mutex.clone(), ukf_mutex.clone());

    let start = std::time::Instant::now();
    let mut pre_u = 0.0;
    loop {
        let x_est = {
            let ukf = ukf_mutex.lock().unwrap();
            ukf.state()
        };
        let x_est = vector![x_est[0], x_est[1], x_est[3], x_est[4]];

        // θの絶対値がpi/2を超えればエラー
        if x_est[2].abs() > std::f64::consts::PI / 2.0 {
            println!("θ is over pi/2");
            break;
        }

        let mut u_n = {
            let u_n = u_n_mutex.lock().unwrap();
            *u_n
        };

        let _ = solve_control_optimization(&x_est, &mut u_n, &mut panoc_cache)
            .expect("Failed to solve");

        if approx_equal(pre_u, u_n[0]) {
            continue;
        }
        pre_u = u_n[0];

        {
            let mut tmp = u_n_mutex.lock().unwrap();
            *tmp = u_n;
        }

        print_con(start, &u_n, &x_est);
    }
}

// MARK: - Dynamics
const A: na::Matrix4<f64> = matrix![
    1.0, DT, 0.0, 0.0;
    0.0, 1.0, -M2 * M2 * G * L * L / D * DT, 0.0;
    0.0, 0.0, 1.0, DT;
    0.0, 0.0, (M1 + M2 + J1 / (R_W * R_W)) / D * M2 * G * L * DT, 1.0
];
const B: na::Vector4<f64> = matrix![
    0.0;
    (M2 * L * L + J2) / D / R_W * KT * DT;
    0.0;
    -M2 * L / D / R_W * KT * DT;
];

// 系ダイナミクスを記述
const M1: f64 = 160e-3;
const R_W: f64 = 50e-3;
const M2: f64 = 2.16 - 2.0 * M1;
const L: f64 = 0.4; // 重心までの距離
const J1: f64 = 2.23e5 * 1e-9; // タイヤの慣性モーメント
const J2: f64 = 1.168e8 * 1e-9; // リポあり
const G: f64 = 9.81;
const KT: f64 = 0.15; // m2006 * 2
const D: f64 = (M1 + 0.5 * M2 + J1 / (R_W * R_W)) * (M2 * L * L + J2) - 0.5 * M2 * M2 * L * L;
// 非線形
fn dynamics_short(x: &na::Vector6<f64>, u: f64, dt: f64) -> na::Vector6<f64> {
    let mut r = *x;
    const D: f64 = (M1 + M2 + J1 / (R_W * R_W)) * (M2 * L * L + J2);
    let d = D - 0.5 * (M2 * L * x[2].cos()).powi(2);
    r[0] += x[1] * dt;
    r[1] += x[2] * dt;
    let term3 = (J2 + M2 * L * L) * (KT * u / R_W + M2 * L * x[4].powi(2) * x[3].sin());
    let term4 = 0.5 * M2 * G * L * L * x[3].sin() * x[3].cos();
    r[2] = (term3 + term4) / d;
    r[3] += x[4] * dt;
    r[4] += x[5] * dt;
    let term1 = (M1 + 0.5 * M2 + J1 / (R_W * R_W)) * M2 * G * L * x[3].sin();
    let term2 = (KT * u / R_W + 0.5 * M2 * L * x[4].powi(2) * x[3].sin()) * M2 * L * x[3].cos();
    r[5] = (term1 - term2) / d;
    r
}

// MARK: - MPC
const M: usize = 4;
fn cost(x: &na::Vector4<f64>, u: &na::SVectorView<f64, N>) -> f64 {
    let f = create_f_matrix!(A, N, M);
    let g = create_g_matrix!(A, B, N, M);
    let q = create_q_matrix!(C, N, M);

    let x_ref = gen_ref(x);
    let x_ref = na::SVectorView::<f64, { M * N }>::from_slice(x_ref.as_slice());
    let left = u.transpose() * g.transpose() * q * g * u;
    let right = 2.0 * (x.transpose() * f.transpose() - x_ref.transpose()) * q * g * u;
    left[0] + right[0]
}
fn grad_cost(x: &na::Vector4<f64>, u: &na::SVectorView<f64, N>) -> na::SVector<f64, N> {
    let f = create_f_matrix!(A, N, M);
    let g = create_g_matrix!(A, B, N, M);
    let q = create_q_matrix!(C, N, M);

    let x_ref = gen_ref(x);
    let x_ref = na::SVectorView::<f64, { M * N }>::from_slice(x_ref.as_slice());
    2.0 * g.transpose() * q * (g * u + f * x - x_ref)
}

fn gen_ref(x: &na::Vector4<f64>) -> na::SMatrix<f64, 4, N> {
    let mut r = na::SMatrix::<f64, 4, N>::zeros();
    for i in 0..N {
        let phase = std::f64::consts::PI * i as f64 / N as f64;
        r[(0, i)] = (x[0] * (1.0 + phase.cos())) / 2.0;
        r[(1, i)] = (-0.75 * x[0]).clamp(-2.0, 2.0) * phase.sin();
        r[(2, i)] = (-0.5 * x[0]).clamp(-0.35, 0.35) * (1.0 * phase.cos()) / 2.0;
        r[(3, i)] = (-0.5 * x[0]).clamp(-1.5, 1.5) * phase.sin();
    }
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
        az / G,                                    // 垂直方向の力 [m/s^2] -> [G]
        ax / G,                                    // 水平方向の力 [m/s^2] -> [G]
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
    let dt_3 = dt.powi(3);
    let q = matrix![
        0.0, 0.0       , 0.0       , 0.0       , 0.0       , 0.0;
        0.0, 0.0       , 0.0       , 0.0       , 0.5 * dt_3, 0.5 * dt_2;
        0.0, 0.0       , 0.0       , 0.5 * dt_3, 1.0 * dt_2, 1.0 * dt;
        0.0, 0.0       , 0.5 * dt_3, 0.0       , 0.5 * dt_3, 0.5 * dt_2;
        0.0, 0.5 * dt_3, 1.0 * dt_2, 0.5 * dt_3, 1.0 * dt_2, 1.0 * dt;
        0.0, 0.5 * dt_2, 1.0 * dt  , 0.5 * dt_2, 1.0 * dt  , 1.0;
    ];
    PHY * q
}

// MARK: - Optimization
fn solve_control_optimization(
    x_est: &na::Vector4<f64>,
    u_n: &mut na::SVector<f64, N>,
    panoc_cache: &mut PANOCCache,
) -> Result<optimization_engine::core::SolverStatus, SolverError> {
    let max_dur: std::time::Duration = std::time::Duration::from_secs_f64(DUR);

    let f = |u: &[f64], c: &mut f64| -> Result<(), SolverError> {
        let u = na::SVectorView::<f64, N>::from_slice(u);
        *c = cost(x_est, &u);
        Ok(())
    };

    let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
        let u = na::SVectorView::<f64, N>::from_slice(u);
        let g = grad_cost(x_est, &u);
        grad.copy_from_slice(g.as_slice());
        Ok(())
    };

    let bounds = constraints::Rectangle::new(Some(&[LIMIT.0]), Some(&[LIMIT.1]));
    let problem = Problem::new(&bounds, df, f);
    let mut panoc = PANOCOptimizer::new(problem, panoc_cache)
        .with_max_iter(usize::MAX)
        .with_max_duration(max_dur);
    panoc.solve(u_n.as_mut_slice())
}

// MARK: - App
fn start_dynamics_thread(
    x_mutex: Arc<Mutex<na::Vector6<f64>>>,
    u_n_mutex: Arc<Mutex<na::SVector<f64, N>>>,
) {
    thread::spawn(move || {
        let start = std::time::Instant::now();
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
            pre = std::time::Instant::now();
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
        let start = std::time::Instant::now();
        let mut pre = start;
        loop {
            let x = {
                // ロックを取得できるまで待機
                let x = x.lock().expect("Failed to lock");
                *x
            };
            let x_obs = sensor(&x);
            // センサの遅延
            thread::sleep(Duration::from_micros(1500));
            let u = {
                let u_n = u_n_mutex.lock().unwrap();
                u_n[0]
            };
            let (x_est, p) = {
                // ロックを取得できるまで待機
                let mut ukf = ukf_mutex.lock().expect("Failed to lock");
                let dt = pre.elapsed().as_secs_f64();
                pre = std::time::Instant::now();
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
fn print_con(start: std::time::Instant, u_n: &na::SVector<f64, N>, x_est: &na::SVector<f64, 4>) {
    print!("\x1b[32mCon:");
    print!("{:5.2} ", start.elapsed().as_secs_f64());
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
    start: std::time::Instant,
    u: f64,
    x: &na::SVector<f64, 6>,
    p: &na::SMatrix<f64, 6, 6>,
) {
    let h = hx(x_est);
    let z = (x_obs - h).abs();
    print!("\x1b[36mRcv:\x1b[m");
    print!("{:5.2} ", start.elapsed().as_secs_f64());
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
        "p:[{:6.2},{:5.2},{:5.2},{:5.2}] ",
        p[(0, 0)],
        p[(1, 1)],
        p[(3, 3)],
        p[(4, 4)],
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
        let mut wtr = csv::Writer::from_path("logs/mpc-ukf.csv").expect("Failed to create file");
        let start = std::time::Instant::now();
        let mut pre_write = start;
        loop {
            // ログの書き込み 100msごと
            if pre_write.elapsed() > Duration::from_millis(100) {
                pre_write = std::time::Instant::now();

                let u = {
                    let u = u_n_mutex.lock().unwrap();
                    u[0]
                };
                let x = {
                    let x = x_mutex.lock().unwrap();
                    *x
                };
                let x_est = {
                    let ukf = ukf_mutex.lock().unwrap();
                    ukf.state()
                };

                write(&mut wtr, start.elapsed().as_secs_f64(), u, x, x_est)
                    .expect("Failed to write");
            }
        }
    });
}
