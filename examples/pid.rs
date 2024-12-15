extern crate nalgebra as na;
use advanced_pid::{PidConfig, PidController, VelPid};
use na::{vector, Vector4};
use std::f64::consts::PI;

const DT: f64 = 1e-3;

fn main() {
    let mut x: Vector4<f64> = vector![-0.5, 0.0, 0.2, 0.0];

    // ログファイルの作成
    let file_path = "logs/pid/pid.csv";
    let mut wtr = csv::Writer::from_path(file_path).expect("file open error");

    let mut pid_theta = VelPid::new(PidConfig::new(6e-1, 4e-1, 5e-3).with_limits(-25.0, 25.0));
    // let mut pid_theta = VelPid::new(PidConfig::new(5.2e-1, 6e-1, 5e-5).with_limits(-25.0, 25.0));

    let now = std::time::Instant::now();
    let mut i = 0;
    while i as f64 * DT < 10.0 {
        let t = i as f64 * DT;
        let p = 0.5;
        let phase = x[0].clamp(-p, p) * PI / p / 2.0;
        let theta_ref = -0.2 * phase.sin().powi(5);
        // let phase = x[0].clamp(-p, p) * PI / p;
        // let theta_ref = -0.15 * (1.0 - phase.cos()) / 2.0 * phase.signum();
        let u = -pid_theta.update(theta_ref, x[2], DT);
        x = dynamics(&x, u);

        // 0.1秒ごとにログを記録
        if i % (0.1 / DT) as i32 == 0 {
            println!(
                "t: {:.2}, r: {:8.5}, u: {:8.3}, x: [{:10.4}, {:6.2}, {:5.2}, {:5.2}]",
                t, theta_ref, u, x[0], x[1], x[2], x[3]
            );

            wtr.write_record(&[
                t.to_string(),
                u.to_string(),
                theta_ref.to_string(),
                x[0].to_string(),
                x[1].to_string(),
                x[2].to_string(),
                x[3].to_string(),
            ])
            .expect("write error");
            wtr.flush().expect("flush error");
        }

        // x[2] が 60度 以上になったら終了
        if x[2].abs() > 60.0f64.to_radians() {
            println!("x[2] is over 60 degrees");
            break;
        }

        i += 1;
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
const D: f64 = (M1 + M2 + J1 / R_W * R_W) * (M2 * L * L + J2) - M2 * M2 * L * L;
fn dynamics(state: &na::Vector4<f64>, u: f64) -> na::Vector4<f64> {
    let mut x = *state;
    x[3] += ((M1 + M2 + J1 / R_W * R_W) / D * M2 * G * L * x[2] - M2 * L / D / R_W * KT * u) * DT;
    x[2] += x[3] * DT;
    x[1] += (-M2 * M2 * G * L * L / D * x[2] + (M2 * L * L + J2) / D / R_W * KT * u) * DT;
    x[0] += x[1] * DT;
    x
}
