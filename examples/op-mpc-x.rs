use optimization_engine::{panoc::*, *};
extern crate nalgebra as na;
use anyhow::{bail, Result};
use rayon::prelude::*;

// 車輪型倒立振子の制御を行う
// 状態変数はθ, θ', x, x'の4つ
// 入力は車輪の電流I
// 状態方程式は以下の通り
// 定数 D = (m1 + m2 + J1 / r^2)(m2 l^2 + J2) - m2^2 l^2
// x'' = m2^2 g l^2 / D θ - (m2 l2^ + J2) / D r Kt I
// θ'' = (m1 + m2 + J1 / r^2) / D m2 g l θ - m2 l / D r Kt I

// 座標軸 x: 前向き, y: 上向き, z: 左向き

// 大きい版
// const R: f64 = 100e-3;
// const M1: f64 = 350e-3;
// const M2: f64 = 10.0
// 本体の慣性モーメント
// const J2: f64 = 0.1157;
// const KT: f64 = 3.0 / 8.0; // m3508

// タイヤ 150 [g]
// タイヤ半径 50 [mm]
const M1: f64 = 150e-3;
const R: f64 = 50e-3;

// 全体重量 2.3 [kg]
const M2: f64 = 2.3 - 2.0 * M1;
// const M2: f64 = 2.3 - 2.0 * M1 + 2.0;

const L: f64 = 0.2474; // 重心までの距離
const J1: f64 = M1 * R * R;
// 本体の慣性モーメント 4.177e7 [g * mm^2] -> [kg * m^2]
// const J2: f64 = M2 * L * L;
// const J2: f64 = 4.177e7 * 1e-9;
// const J2: f64 = 0.1;
const J2: f64 = 0.2;
// const J2: f64 = 0.4;
// const J2: f64 = 0.6;
// const J2: f64 = 0.8;
// const J2: f64 = 1.2;

const G: f64 = 9.81;
const KT: f64 = 0.15; // m2006
const D: f64 = (M1 + M2 + J1 / R * R) * (M2 * L * L + J2) - M2 * M2 * L * L;

const T: f64 = 0.5;
const N: usize = 50;
// const T: f64 = 0.5;
// const N: usize = 100;
// const T: f64 = 0.5;
// const N: usize = 200;
// const T: f64 = 0.75;
// const N: usize = 75;
// const T: f64 = 0.75;
// const N: usize = 150;
// const T: f64 = 1.0;
// const N: usize = 100;
const DT: f64 = T / N as f64;

// 逐次実行版
fn dynamics(x: &mut [f64; 4], u: f64) {
    x[3] += ((M1 + M2 + J1 / R * R) / D * M2 * G * L * x[2] - M2 * L / D / R * KT * u) * DT;
    x[2] += x[3] * DT;
    x[1] += (-M2 * M2 * G * L * L / D * x[2] + (M2 * L * L + J2) / D / R * KT * u) * DT;
    x[0] += x[1] * DT;
}

#[allow(dead_code)]
fn dynamics_cpy(mut x: [f64; 4], u: f64) -> [f64; 4] {
    dynamics(&mut x, u);
    x
}

// const GAIN: [f64; 5] = [0e-6, 0e-6, 16e-6, 0e-6, 0e-6];
// const GAIN: [f64; 5] = [0e-6, 0.001e-6, 16e-6, 0e-6, 0e-6];
// const GAIN: [f64; 5] = [0e-6, 0e-6, 16e-6, 0e-6, 0e-6];
const GAIN: [f64; 5] = [0.0, 9.2, 16.0, 0.5, 0.0];
// const GAIN: [f64; 5] = [0.0, 0.01, 16.0, 0.005, 0.0005];
// const GAIN: [f64; 5] = [0.3, 0.0, 0.9, 0.1, 0.0];
// const GAIN: [f64; 5] = [0.3, 0.3, 0.9, 0.1, 0.00005];

// 参照経路(x_ref)を生成する
fn planning_err(x: &[f64; 4]) -> [f64; 4] {
    let x_g = x[0] + x[2] * L;
    let x_g_ref = (0.0 - x_g).clamp(-1.5, 1.5);
    let x_g_err = x_g_ref - x_g;
    let x_g_dot = x[1] + x[3] * L;
    let x_g_dot_ref = (1.5 * x_g_err).clamp(-1.5, 1.5);
    let x_g_dot_err = x_g_dot_ref - x_g_dot;
    // let theta_ref = 0.0;
    let theta_ref = (0.5 * x_g_dot_err).clamp(-0.3, 0.3);
    // let theta_ref = (0.5 * x_g_dot_err).clamp(-0.4, 0.4);
    let theta_err = theta_ref - x[2];
    let theta_dot_ref = 0.0;
    let theta_dot_err = theta_dot_ref - x[3];

    // plan: [x_g_ref, x_g_dot_ref, theta_ref, theta_dot_ref]
    [x_g_err, x_g_dot_err, theta_err, theta_dot_err]
}

// 評価関数は以下の通り
// J = \sum (x - x_ref)^2 + (θ-θ_ref)^2 + u^2
fn cost(mut x: [f64; 4], u: &[f64]) -> f64 {
    let mut c = 0.0;

    for e in u.iter() {
        dynamics(&mut x, *e);
        let x_err = planning_err(&x);
        // 追従誤差のコスト
        c += GAIN[0] * x_err[0].powi(2)
            + GAIN[1] * x_err[1].powi(4)
            + GAIN[2] * x_err[2].powi(4)
            + GAIN[3] * x_err[3].powi(4)
            + GAIN[4] * e.powi(2);
        // // 0.43[rad] = 25[deg] を超えるとコスト
        // c += 1e-6 * (x[2].cosh() - 1.1).max(0.0);
        // 0.62[rad] = 35.5[deg] を超えるとコスト
        // c += 1e-6 * x[2].cosh() - 1.2;
        c += (x[2].cosh() - 1.2).max(0.0);
    }
    c
}

// コスト関数の勾配を数値微分によって求める(並列化版)
// 1. 数値微分の基礎となるコストを計算
// 2. 初期値xに運動方程式を適用し、kステップ先の状態を求める
// 3. kステップ先の状態と、入力に対して微小な変化を加えてコストを計算
// 4. 3.で求めたコストと1.のコストの差分を取り、微分を求める
fn grad_cost(x: [f64; 4], u: &[f64; N], grad: &mut [f64]) {
    const EPS: f64 = 0.001;
    (0..N)
        .map(|i| dynamics_cpy(x, u[i]))
        .collect::<Vec<_>>()
        .par_iter_mut()
        .zip(grad)
        .enumerate()
        .for_each(|(i, (x, e))| {
            let mut u_cpy = *u;
            u_cpy[i] += EPS;
            let posi = cost(*x, &u_cpy);
            u_cpy[i] -= 2.0 * EPS;
            let nega = cost(*x, &u_cpy);
            let g = (posi - nega) / (2.0 * EPS);
            *e = g;
            // *e = g.clamp(-1e300, 1e300);
            // *e = g.clamp(f64::MIN, f64::MAX);
        });
}

fn main() -> Result<()> {
    let file_path = "logs/op-mpc-x.csv";
    let mut wtr = csv::Writer::from_path(file_path)?;
    // let mut wtr = csv::Writer::from_writer(std::io::stdout());

    let tolerance = 1e-6;
    let lbfgs_memory = 20;
    let max_iters = usize::MAX;
    let max_dur = std::time::Duration::from_secs_f64(DT);
    // let max_dur = std::time::Duration::from_millis(10);
    let mut panoc_cache = PANOCCache::new(N, tolerance, lbfgs_memory);

    let mut u = [0.0; N];

    // let mut x = [0.1, 0.0, -0.2, 0.0];
    let mut x = [3.0, 0.0, -0.7, 0.0];
    // let mut x = [0.0, 0.0, 0.1, 0.0];
    // let mut x = [0.0, 0.5, 0.0, 0.0];
    // let mut x = [-1.0, 0.0, 0.3, 0.0];

    const MAX_ITERS: usize = (10.0 / DT) as usize;
    for i in 0..MAX_ITERS + 1 {
        let f = |u: &[f64], c: &mut f64| -> Result<(), SolverError> {
            *c = cost(x, u);
            Ok(())
        };

        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            let u: [f64; N] = u.try_into().expect("slice with incorrect length");
            grad_cost(x, &u, grad);
            Ok(())
        };

        // define the bounds at every iteration
        // let bounds = constraints::NoConstraints::new();
        let bounds = constraints::Rectangle::new(Some(&[-30.0]), Some(&[30.0]));

        // the problem definition is updated at every iteration
        let problem = Problem::new(&bounds, df, f);

        // updated instance of the solver
        let mut panoc = PANOCOptimizer::new(problem, &mut panoc_cache)
            .with_max_iter(max_iters)
            .with_max_duration(max_dur);

        // let status = panoc.solve(&mut u).map_err(|e| anyhow!("{:?}", e))?;
        let _status = loop {
            match panoc.solve(&mut u) {
                Ok(status) => {
                    if status.iterations() == 0 || u[0].abs() >= 30.0 {
                        println!(
                            "\x1b[31mIncorrect States: {:?}, {} -> retry\x1b[0m",
                            status, u[0]
                        );
                        u.iter_mut().for_each(|e| *e = 0.0);
                        continue;
                    }
                    break status;
                }
                Err(e) => {
                    println!("\x1b[31mSolverError: {:?} -> retry\x1b[0m", e);
                    u.iter_mut().for_each(|e| *e = 0.0);
                    continue;
                }
            }
        };

        let mut x_est = x;
        for e in u.iter() {
            dynamics(&mut x_est, *e);
        }

        dynamics(&mut x, u[0]);
        // let t = i as f64 * DT;
        // let x_g = x[0] + x[2] * L;
        // let x_g_dot = x[1] + x[3] * L;
        // println!(
        //     "{t:5.3}, {i:4}/{MAX_ITERS}, {:4}, {:#.6}, {:?}, {:5.2}, {:5.2}",
        //     status.iterations(),
        //     u[0],
        //     x,
        //     x_g,
        //     x_g_dot
        // );

        print!("{i:4}/{MAX_ITERS}, {:7.2}, ", u[0]);
        print!(
            "act: ({:7.2},{:7.2},{:7.2},{:7.2}) ",
            x[0], x[1], x[2], x[3]
        );
        print!(
            "est: ({:7.2},{:7.2},{:7.2},{:7.2}) ",
            x_est[0], x_est[1], x_est[2], x_est[3]
        );
        println!();

        wtr.write_record(&[
            (i as f64 * DT).to_string(),
            u[0].to_string(),
            x[0].to_string(),
            x[1].to_string(),
            x[2].to_string(),
            x[3].to_string(),
            x_est[0].to_string(),
            x_est[1].to_string(),
            x_est[2].to_string(),
            x_est[3].to_string(),
        ])?;
        wtr.flush()?;

        // x[2]の絶対値がpi/2を超えればエラー
        if x[2].abs() > std::f64::consts::PI / 2.0 {
            bail!("Error: x[2] = {} > PI / 2", x[2])
        }
    }

    wtr.flush()?;
    Ok(())
}
