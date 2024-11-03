extern crate nalgebra as na;
use mpc::packet::{Control, Sensor2 as Sensor};
use mpc::ukf2::UnscentedKalmanFilter;
use na::{matrix, vector};
use optimization_engine::{panoc::*, *};
use std::f64::consts::PI;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// 予測ホライゾン
const T: f64 = 0.8;
const N: usize = 16;
const DT: f64 = T / N as f64;

fn main() {
    let mut port = serialport::new("/dev/ttyUSB0", 115_200)
        .timeout(Duration::from_millis(10))
        .open()
        .expect("Failed to open port");
    let mut reader = BufReader::new(port.try_clone().unwrap());

    let tolerance = 1e-6;
    let lbfgs_memory = 20;
    let max_iters = usize::MAX;
    let max_dur = std::time::Duration::from_secs_f64(DT);
    let mut panoc_cache = PANOCCache::new(N, tolerance, lbfgs_memory);

    let init_u_n = na::SVector::<f64, N>::zeros();
    let u_n_mutex = Arc::new(Mutex::new(init_u_n));
    let u_n_mutex1 = u_n_mutex.clone();
    let u_n_mutex2 = u_n_mutex.clone();
    let init_x = vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let ukf_mutex = init_ukf(&init_x);
    let ukf_mutex1 = ukf_mutex.clone();

    thread::spawn(move || {
        // データが読み込まれるまで待機
        let start = std::time::Instant::now();
        let mut pre = start;
        loop {
            if let Some(s) = read(&mut reader) {
                let x_obs = s.into();
                let u = {
                    let u_n = u_n_mutex1.lock().unwrap();
                    u_n[0]
                };
                let (x_est, p) = {
                    // ロックを取得できるまで待機
                    let mut ukf = ukf_mutex.lock().expect("Failed to lock");
                    let dt = pre.elapsed().as_secs_f64();
                    pre = std::time::Instant::now();
                    let fx = |x: &_, u| dynamics_short(x, u, dt);
                    ukf.predict(u, fx);
                    ukf.update(&x_obs, hx);
                    (ukf.state(), ukf.covariance())
                };
                print!("\x1b[36mRcv: \x1b[m");
                print!("t: {:5.2} ", start.elapsed().as_secs_f64());
                print!(
                    "est: [{:6.2}, {:5.2}, {:5.2}, {:5.2}] ",
                    x_est[0], x_est[1], x_est[3], x_est[4]
                );
                print!(
                    "p: [{:6.2}, {:5.2}, {:5.2}, {:5.2}] ",
                    p[(0, 0)],
                    p[(1, 1)],
                    p[(3, 3)],
                    p[(4, 4)]
                );
                print!(
                    "obs: [{:6.2}, {:5.2}, {:5.2}, {:5.2}, {:5.2}] ",
                    x_obs[0], x_obs[1], x_obs[2], x_obs[3], x_obs[4]
                );
                print!("u: {:8.3} ", u);
                println!();
            }
        }
    });

    let start = std::time::Instant::now();
    let mut pre = start;
    loop {
        let x_est = {
            let ukf = ukf_mutex1.lock().unwrap();
            ukf.state()
        };

        // x[2]の絶対値がpi/2を超えればエラー
        if x_est[2].abs() > std::f64::consts::PI / 2.0 {
            println!("x[2] is over pi/2");
            break;
        }

        let f = |u: &[f64], c: &mut f64| -> Result<(), SolverError> {
            let u = na::SVectorView::<f64, N>::from_slice(u);
            *c = cost(&x_est, &u);
            Ok(())
        };

        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            let u = na::SVectorView::<f64, N>::from_slice(u);
            let g = grad_cost(&x_est, &u);
            grad.copy_from_slice(g.as_slice());
            Ok(())
        };

        let mut u = {
            let u = u_n_mutex2.lock().unwrap();
            *u
        };

        let bounds = constraints::Rectangle::new(Some(&[-10.0]), Some(&[10.0]));
        let problem = Problem::new(&bounds, df, f);
        let mut panoc = PANOCOptimizer::new(problem, &mut panoc_cache)
            .with_max_iter(max_iters)
            .with_max_duration(max_dur);
        let _status = panoc.solve(u.as_mut_slice()).expect("Failed to solve");

        if pre.elapsed().as_secs_f64() < DT {
            thread::sleep(Duration::from_secs_f64(DT - pre.elapsed().as_secs_f64()));
        }
        pre = std::time::Instant::now();

        {
            let mut tmp = u_n_mutex2.lock().unwrap();
            *tmp = u;
        }

        let c = Control::from_current(u[0]);
        write(&mut port, &c);

        print!("\x1b[32mCon: \x1b[m");
        print!("t: {:5.2} ", start.elapsed().as_secs_f64());
        print!(
            "est: [{:6.2}, {:5.2}, {:5.2}, {:5.2}] ",
            x_est[0], x_est[1], x_est[3], x_est[4]
        );
        print!("u: {:8.3} ", u[0]);
        println!();
    }
}

macro_rules! create_a_matrix {
    ($a:expr, $n:expr) => {{
        let mut a = na::SMatrix::<f64, { 4 * $n }, 4>::zeros();
        for i in 0..$n {
            a.fixed_view_mut::<4, 4>(4 * i, 0)
                .copy_from(&$a.pow((i + 1) as u32));
        }
        a
    }};
}
macro_rules! create_g_matrix {
    ($a:expr, $b:expr, $n:expr) => {{
        let mut g = na::SMatrix::<f64, { 4 * $n }, $n>::zeros();
        for i in 0..$n {
            for j in 0..=i {
                g.fixed_view_mut::<4, 1>(4 * i, j)
                    .copy_from(&(A.pow((i - j) as u32) * B));
            }
        }
        g
    }};
}
macro_rules! create_q_matrix {
    ($c:expr, $n:expr) => {{
        let mut q = na::SMatrix::<f64, { 4 * $n }, { 4 * $n }>::zeros();
        for i in 0..$n {
            q.fixed_view_mut::<4, 4>(4 * i, 4 * i).copy_from(&$c);
        }
        q
    }};
}

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
const C: na::Matrix4<f64> = matrix![
    5.0, 0.0, 0.0, 0.0;
    0.0, 5.0, 0.0, 0.0;
    0.0, 0.0, 1.0, 0.0;
    0.0, 0.0, 0.0, 1.0
];

// 系ダイナミクスを記述
const M1: f64 = 150e-3;
const R_W: f64 = 50e-3;
const M2: f64 = 2.3 - 2.0 * M1 + 2.0;
const L: f64 = 0.2474; // 重心までの距離
const J1: f64 = M1 * R_W * R_W;
const J2: f64 = 0.2;
const G: f64 = 9.81;
const KT: f64 = 0.15; // m2006
const D: f64 = (M1 + M2 + J1 / (R_W * R_W)) * (M2 * L * L + J2) - M2 * M2 * L * L;
// 非線形
fn dynamics_short(x: &na::Vector6<f64>, u: f64, dt: f64) -> na::Vector6<f64> {
    let mut r = *x;
    const D: f64 = (M1 + M2 + J1 / (R_W * R_W)) * (M2 * L * L + J2);
    let d = D - M2 * M2 * L * L * x[3].cos() * x[3].cos();
    let term1 = (M1 + M2 + J1 / (R_W * R_W)) * M2 * G * L * x[3].sin();
    let term2 = (KT * u / R_W + M2 * L * x[4].powi(2) * x[3].sin()) * M2 * L * x[3].cos();
    r[5] = (term1 - term2) / d;
    r[4] += x[5] * dt;
    r[3] += x[4] * dt;
    let term3 = (J2 + M2 * L * L) * (KT * u / R_W + M2 * L * x[4].powi(2) * x[3].sin());
    let term4 = M2 * G * L * L * x[3].sin() * x[3].cos();
    r[2] = (term3 + term4) / d;
    r[1] += x[2] * dt;
    r[0] += x[1] * dt;
    r
}

fn cost<S>(x: &na::Vector6<f64>, u: &na::Vector<f64, na::Const<N>, S>) -> f64
where
    S: na::Storage<f64, na::Const<N>>,
{
    let a: na::SMatrix<f64, { 4 * N }, 4> = create_a_matrix!(A, N);
    let g: na::SMatrix<f64, { 4 * N }, N> = create_g_matrix!(A, B, N);
    let q = create_q_matrix!(C, N);

    let x = vector![x[0], x[1], x[3], x[4]];
    let x_ref = gen_ref(&x);
    let x_ref = na::SVectorView::<f64, { 4 * N }>::from_slice(x_ref.as_slice());
    let left = u.transpose() * g.transpose() * q * g * u;
    let right = 2.0 * (x.transpose() * a.transpose() - x_ref.transpose()) * q * g * u;
    left[0] + right[0]
}

fn grad_cost<S>(x: &na::Vector6<f64>, u: &na::Vector<f64, na::Const<N>, S>) -> na::SVector<f64, N>
where
    S: na::Storage<f64, na::Const<N>>,
{
    let a: na::SMatrix<f64, { 4 * N }, 4> = create_a_matrix!(A, N);
    let g: na::SMatrix<f64, { 4 * N }, N> = create_g_matrix!(A, B, N);
    let q = create_q_matrix!(C, N);

    let x = vector![x[0], x[1], x[3], x[4]];
    let x_ref = gen_ref(&x);
    let x_ref = na::SVectorView::<f64, { 4 * N }>::from_slice(x_ref.as_slice());
    2.0 * g.transpose() * q * (g * u + a * x - x_ref)
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

fn init_ukf(init: &na::Vector6<f64>) -> Arc<Mutex<UnscentedKalmanFilter>> {
    let p = matrix![
        10.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 10.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 10.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 10.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 10.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.0, 10.0;
    ];
    let q = matrix![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
        0.0, 0.0, 0.0, 0.0, 1.0, 1e2;
        0.0, 0.0, 0.0, 1.0, 1e2, 1e4;
    ];
    let r = matrix![
        50.0, 0.0, 0.0, 0.0, 0.0;
        0.0, 50.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.5, 0.0, 0.0;
        0.0, 0.0, 0.0, 50.0, 0.0;
        0.0, 0.0, 0.0, 0.0, 50.0;
    ];
    let obj = UnscentedKalmanFilter::new(*init, p, q, r);
    Arc::new(Mutex::new(obj))
}

fn hx(state: &na::Vector6<f64>) -> na::Vector5<f64> {
    let v = M2 * G * state[3].cos() + M2 * state[2] * state[3].sin() - M2 * L * state[4].powi(2);
    let h = -M2 * G * state[3].sin() + M2 * state[2] * state[3].cos() + M2 * L * state[5];
    vector![
        60.0 / (2.0 * PI * R_W) * state[1], // 駆動輪のオドメトリ [m/s] -> [rpm]
        60.0 / (2.0 * PI * R_W) * state[1], // 駆動輪のオドメトリ [m/s] -> [rpm]
        state[4].to_degrees(),              // 角速度 [rad/s] -> [deg/s]
        v / G,                              // 垂直方向の力 [N] -> [G]
        h / G,                              // 水平方向の力 [N] -> [G]
    ]
}

fn write(port: &mut Box<dyn serialport::SerialPort>, c: &Control) {
    let cobs = c.as_cobs();
    port.write_all(&cobs).expect("Write failed!");
}
// UARTを受取り、mpscに送信する
fn read(reader: &mut BufReader<Box<dyn serialport::SerialPort>>) -> Option<Sensor> {
    let mut buf = Vec::new();
    let len = reader.read_until(0x00, &mut buf).ok()?;
    if len >= Sensor::BUF_SIZE {
        let data = buf[(len - Sensor::BUF_SIZE)..len].try_into().ok()?;
        Sensor::from_cobs(&data)
    } else {
        None
    }
}
