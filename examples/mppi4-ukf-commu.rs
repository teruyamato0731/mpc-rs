extern crate nalgebra as na;
use mpc::mppi::Mppi;
use mpc::packet::{Control, Sensor3 as Sensor};
use mpc::ukf2::UnscentedKalmanFilter;
use na::{matrix, vector};
use std::f64::consts::PI;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// MARK: - Constants
// 予測ホライゾン
const T: f64 = 0.9;
const N: usize = 30;
const DT: f64 = T / N as f64;

// 制御ホライゾン
const K: usize = 15e5 as usize;
const LAMBDA: f64 = 0.5;
const R_U: f64 = 5.0;

// 制約
const LIMIT: (f64, f64) = (-10.0, 10.0);

// UKF
const PHY: na::Vector3<f64> = vector![50.0, 50.0, 10.0];
const R: na::SVector<f64, 5> = vector![200.0, 200.0, 20.0, 0.5, 0.5];

const DEBUG: bool = false;

// MARK: - Main
fn main() {
    let mut port = serialport::new("/dev/ttyUSB0", 115_200)
        .timeout(Duration::from_millis(10))
        .open()
        .expect("Failed to open port");
    let reader = BufReader::new(port.try_clone().unwrap());

    let init_u_n = na::SVector::<f64, N>::zeros();
    let u_n_mutex = Arc::new(Mutex::new(init_u_n));
    let mut mppi = Mppi::<N, K>::new(dynamics, cost, LAMBDA, R_U, LIMIT);
    let init_x = vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let ukf_mutex = init_ukf(init_x);

    start_ukf_thread(reader, u_n_mutex.clone(), ukf_mutex.clone());
    start_logging_thread(u_n_mutex.clone(), ukf_mutex.clone());

    let mut pre_u = 0.0;
    let start = Instant::now();
    loop {
        let x_est = {
            let ukf = ukf_mutex.lock().expect("Failed to lock");
            ukf.state()
        };

        // θの絶対値がpi/2を超えればエラー
        if x_est[3].abs() > std::f64::consts::PI / 2.0 {
            println!("x[2] is over pi/2");
            println!(
                "x: [{:6.2}, {:5.2}, {:5.2}, {:5.2}, {:5.2}, {:5.2}] ",
                x_est[0], x_est[1], x_est[2], x_est[3], x_est[4], x_est[5]
            );
            println!("elapsed: {:.2} sec", start.elapsed().as_secs_f64());
            break;
        }

        let x_est = vector![x_est[0], x_est[1], x_est[3], x_est[4]];
        let mut u_n = {
            let u = u_n_mutex.lock().unwrap();
            *u
        };

        if let Ok(u) = mppi.compute(&x_est, &u_n) {
            u_n = u;
        } else {
            u_n = na::SVector::<f64, N>::zeros();
        }

        if approx_equal(pre_u, u_n[0]) {
            continue;
        }
        pre_u = u_n[0];

        if DEBUG {
            u_n[0] = 0.0;
        }

        {
            let mut tmp = u_n_mutex.lock().unwrap();
            *tmp = u_n;
        }

        let c = Control::from_current(u_n[0]);
        write(&mut port, &c);

        print!("\x1b[32mCon: \x1b[m");
        print!("t: {:5.2} ", start.elapsed().as_secs_f64());
        print!(
            "est: [{:6.2}, {:5.2}, {:4.0}, {:4.0}] ",
            x_est[0],
            x_est[1],
            x_est[2].to_degrees(),
            x_est[3].to_degrees()
        );
        print!("u: {:6.2} ", u_n[0]);
        println!();
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
// 非線形
fn dynamics_short(x: &na::Vector6<f64>, u: f64, dt: f64) -> na::Vector6<f64> {
    let mut r = *x;
    let d = D1 - (M2 * L * x[2].cos()).powi(2);
    r[0] += x[1] * dt;
    r[1] += x[2] * dt;
    let term1 = (M2 * L * L + J2) * M2 * L / d * x[4].powi(2) * x[3].sin();
    let term2 = -(M2 * L).powi(2) * G / d * x[3].sin() * x[3].cos();
    let term3 = 2.0 * (M2 * L * L + J2) / (d * R_W) * KT * u;
    r[2] = term1 + term2 + term3;
    r[3] += x[4] * dt;
    r[4] += x[5] * dt;
    let term1 = -(M2 * L).powi(2) / d * x[4].powi(2) * x[3].sin() * x[3].cos();
    let term2 = M2 * G * L * (2.0 * M1 + M2 + 2.0 * J1 / (R_W * R_W)) / d * x[3].sin();
    let term3 = -2.0 * M2 * L / (d * R_W) * KT * u * x[3].cos();
    r[5] = term1 + term2 + term3;
    r
}
fn dynamics(x: &na::Vector4<f64>, u: f64) -> na::Vector4<f64> {
    let r = dynamics_short(&vector![x[0], x[1], x[2], 0.0, x[3], 0.0], u, DT);
    vector![r[0], r[1], r[2], r[4]]
}

// MARK: - MPPI
fn cost(x: &na::Vector4<f64>) -> f64 {
    let x_clamped = x[0].clamp(-2.0, 2.0);
    let term1 = 2.0 * x_clamped.powi(2);
    let term2 = 3.0 * (x[1] + 2.0 * x_clamped).clamp(-5.0, 5.0).powi(2);
    let term3 = 5.0 * (x[2] + 0.35 * x[0].clamp(-0.75, 0.75)).powi(2);
    let term4 = 1.2 * x[3].powi(2);
    term1 + term2 + term3 + term4
}

// MARK: - UKF
fn init_ukf(init: na::Vector6<f64>) -> Arc<Mutex<UnscentedKalmanFilter>> {
    let p = na::SMatrix::<f64, 6, 6>::identity() * 10.0;
    let r = na::SMatrix::<f64, 5, 5>::from_diagonal(&R);
    let q = gen_q(DT);
    let obj = UnscentedKalmanFilter::new(init, p, q, r);
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
fn gen_r(enable: u8) -> na::SMatrix<f64, 5, 5> {
    let mut r = R;
    for i in 0..5 {
        if (enable & (1 << i)) == 0 {
            r[i] = 1e6;
        }
    }
    na::SMatrix::<f64, 5, 5>::from_diagonal(&r)
}

// MARK: - UART
fn write(port: &mut Box<dyn serialport::SerialPort>, c: &Control) {
    let cobs = c.as_cobs();
    port.write_all(&cobs).expect("Write failed!");
}
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

// MARK: - App
fn start_ukf_thread(
    mut reader: BufReader<Box<dyn serialport::SerialPort>>,
    u_n_mutex: Arc<Mutex<na::SVector<f64, N>>>,
    ukf_mutex: Arc<Mutex<UnscentedKalmanFilter>>,
) {
    thread::spawn(move || {
        // データが読み込まれるまで待機
        let start = std::time::Instant::now();
        let mut pre = start;
        loop {
            if let Some(s) = read(&mut reader) {
                let (enable, x_obs) = Sensor::parse(s);
                let u = {
                    let u_n = u_n_mutex.lock().unwrap();
                    u_n[0]
                };
                let (x_est, p) = {
                    // ロックを取得できるまで待機
                    let mut ukf = ukf_mutex.lock().expect("Failed to lock");
                    let dt = pre.elapsed().as_secs_f64();
                    pre = std::time::Instant::now();
                    let fx = |x: &_, u| dynamics_short(x, u, dt);
                    let q = gen_q(dt);
                    ukf.set_q(q);
                    let r = gen_r(enable);
                    ukf.set_r(r);
                    ukf.predict(u, fx);
                    let hx = |state: &_| {
                        // enable bit が 0 なら 0 にする
                        let mut obs = hx(state);

                        for i in 0..5 {
                            if (enable & (1 << i)) == 0 {
                                obs[i] = 0.0;
                            }
                        }
                        obs
                    };
                    ukf.update(&x_obs, hx);
                    (ukf.state(), ukf.covariance())
                };
                print!("\x1b[36mRcv: \x1b[m");
                print!("t: {:5.2} ", start.elapsed().as_secs_f64());
                print!(
                    "est: [{:6.2}, {:5.2}, {:4.0}, {:4.0}] ",
                    x_est[0],
                    x_est[1],
                    x_est[3].to_degrees(),
                    x_est[4].to_degrees()
                );
                print!("u: {:6.2} ", u);
                print!(
                    "p: [{:6.2},{:6.2},{:6.2},{:6.2},{:6.2},{:6.2}] ",
                    p[(0, 0)],
                    p[(1, 1)],
                    p[(2, 2)],
                    p[(3, 3)],
                    p[(4, 4)],
                    p[(5, 5)],
                );
                print_obs(enable, &x_obs);
                println!();
            }
        }
    });
}

fn approx_equal(a: f64, b: f64) -> bool {
    let epsilon = 1e-2;
    (a - b).abs() < epsilon
}

// MARK: - Print
macro_rules! print_obs {
    ($fmt:expr, $x_obs:expr, $enable:expr) => {
        if $enable != 0 {
            print!($fmt, $x_obs);
        } else {
            print!("   -   ");
        }
    };
}

fn print_obs(enable: u8, x_obs: &na::Vector5<f64>) {
    print!("obs: [");
    print_obs!("{:6.0} ", x_obs[0], enable & 0b00001);
    print_obs!("{:6.0} ", x_obs[1], enable & 0b00010);
    print_obs!("{:6.0} ", x_obs[2], enable & 0b00100);
    print_obs!("{:6.2} ", x_obs[3], enable & 0b01000);
    print_obs!("{:6.2} ", x_obs[4], enable & 0b10000);
    print!("] ");
}

// MARK: - Logging
fn start_logging_thread(
    u_n_mutex: Arc<Mutex<na::SVector<f64, N>>>,
    ukf_mutex: Arc<Mutex<UnscentedKalmanFilter>>,
) {
    thread::spawn(move || {
        let mut wtr =
            csv::Writer::from_path("logs/mpc-ukf-com.csv").expect("Failed to create file");
        let start = std::time::Instant::now();
        let mut pre_write = start;
        loop {
            // ログの書き込み 10msごと
            if pre_write.elapsed() > Duration::from_millis(10) {
                pre_write = std::time::Instant::now();

                let u = {
                    let u = u_n_mutex.lock().unwrap();
                    u[0]
                };
                let (x_est, p) = {
                    let ukf = ukf_mutex.lock().unwrap();
                    (ukf.state(), ukf.covariance())
                };

                wtr.write_record(&[
                    start.elapsed().as_secs_f64().to_string(),
                    u.to_string(),
                    x_est[0].to_string(),
                    x_est[1].to_string(),
                    x_est[2].to_string(),
                    x_est[3].to_string(),
                    x_est[4].to_string(),
                    x_est[5].to_string(),
                    p[(0, 0)].to_string(),
                    p[(1, 1)].to_string(),
                    p[(2, 2)].to_string(),
                    p[(3, 3)].to_string(),
                    p[(4, 4)].to_string(),
                    p[(5, 5)].to_string(),
                ])
                .expect("Failed to write record");
                wtr.flush().expect("Failed to flush");
            }
        }
    });
}
