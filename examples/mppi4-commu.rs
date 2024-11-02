extern crate nalgebra as na;
use mpc::mppi::Mppi;
use mpc::packet::{Control, State};
use std::io::{BufRead, BufReader};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

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
    let mut port = serialport::new("/dev/ttyUSB0", 115_200)
        .timeout(Duration::from_millis(10))
        .open()
        .expect("Failed to open port");
    let mut reader = BufReader::new(port.try_clone().unwrap());

    let mut x = na::Vector4::new(0.5, 0.0, 0.1, 0.0);
    let mut u_n = na::SVector::<f64, N>::zeros();

    let mut mppi = Mppi::<N, K>::new(dynamics, cost, LAMBDA, R, LIMIT);

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        // データが読み込まれるまで待機
        loop {
            if let Some(s) = read(&mut reader) {
                tx.send(s).unwrap();
            }
        }
    });

    let mut pre = std::time::Instant::now();
    loop {
        // データが読み込まれるまで待機
        if let Ok(received) = rx.try_recv() {
            println!("{:?}", received);
            x = received.into();
            // θ が 60度 以上になったら終了
            if x[2] > 60.0f64.to_radians() {
                println!("x[2] is over 60 degrees");
                break;
            }
        }

        // 0.1sごとに制御信号を送信
        if pre.elapsed() >= Duration::from_millis(100) {
            u_n = mppi.compute(&x, &u_n).unwrap();
            let c = Control::from_current(u_n[0]);
            write(&mut port, &c);
            pre = std::time::Instant::now();
        }
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
fn write(port: &mut Box<dyn serialport::SerialPort>, c: &Control) {
    let cobs = c.as_cobs();
    port.write_all(&cobs).expect("Write failed!");
}
// UARTを受取り、mpscに送信する
fn read(reader: &mut BufReader<Box<dyn serialport::SerialPort>>) -> Option<State> {
    let mut buf = Vec::new();
    let len = reader.read_until(0x00, &mut buf).ok()?;
    if len >= State::BUF_SIZE {
        let data = buf[(len - State::BUF_SIZE)..len].try_into().ok()?;
        State::from_cobs(&data)
    } else {
        None
    }
}
