extern crate nalgebra as na;

use anyhow::{bail, Result};
use na::{matrix, vector};
use optimization_engine::{panoc::*, *};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

// 車輪型倒立振子の制御を行う
// 状態変数はθ, θ', x, x'の4つ
// 入力は車輪の電流I
// 状態方程式は以下の通り
// 定数 D = (m1 + m2 + J1 / r^2)(m2 l^2 + J2) - m2^2 l^2
// x'' = m2^2 g l^2 / D θ - (m2 l2^ + J2) / D r Kt I
// θ'' = (m1 + m2 + J1 / r^2) / D m2 g l θ - m2 l / D r Kt I

const M1: f64 = 150e-3;
const R_W: f64 = 50e-3;
const M2: f64 = 2.3 - 2.0 * M1;
// const M2: f64 = 2.3 - 2.0 * M1 + 2.0;
const L: f64 = 0.2474; // 重心までの距離
const J1: f64 = M1 * R_W * R_W;
const J2: f64 = 1.2;
// const J2: f64 = 0.1;
const G: f64 = 9.81;
const KT: f64 = 0.15; // m2006
const D: f64 = (M1 + M2 + J1 / (R_W * R_W)) * (M2 * L * L + J2) - M2 * M2 * L * L;

const T: f64 = 0.5;
const N: usize = 10;
// const T: f64 = 1.0;
// const N: usize = 50;
// const T: f64 = 0.5;
// const N: usize = 50;
const DT: f64 = T / N as f64;

// LAMBDA = α^2 * (n + κ) - n
const S: f64 = 4.0;
const ALPHA: f64 = 1e-3;
const BETA: f64 = 2.0;
const KAPPA: f64 = 3.0 - S;
const C: f64 = ALPHA * ALPHA * (S + KAPPA); // C := N + LAMBDA
const LAMBDA: f64 = C - S;

const Q: na::Matrix4<f64> = matrix![
    0.0, 0.0, 0.0, 0.0;
    0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 0.0;
    0.0, 0.0, 0.0, 1.0;
];
const R: na::Matrix2<f64> = matrix![
    0.75, 0.75;
    0.75, 0.75;
];
fn sigma_weight() -> (na::SVector<f64, 9>, na::SVector<f64, 9>) {
    let mut wm = na::SVector::<f64, 9>::from_element(1.0 / (2.0 * C));
    let mut wc = na::SVector::<f64, 9>::from_element(1.0 / (2.0 * C));
    wm[0] = LAMBDA / C;
    wc[0] = LAMBDA / C + 1.0 - ALPHA.powi(2) + BETA;
    (wm, wc)
}

fn sigma_points(x: &na::Vector4<f64>, p: &na::Matrix4<f64>) -> na::SMatrix<f64, 4, 9> {
    let l = (C * p).cholesky().expect("Cholesky fail").l();
    na::SMatrix::<f64, 4, 9>::from_columns(&[
        *x,
        *x + l.column(0),
        *x - l.column(0),
        *x + l.column(1),
        *x - l.column(1),
        *x + l.column(2),
        *x - l.column(2),
        *x + l.column(3),
        *x - l.column(3),
    ])
}

fn unscented_transform<const S: usize>(
    sigmas: &na::SMatrix<f64, S, 9>,
    wm: &na::SVector<f64, 9>,
    wc: &na::SVector<f64, 9>,
    cov: &na::SMatrix<f64, S, S>,
) -> (na::SVector<f64, S>, na::SMatrix<f64, S, S>) {
    let x = sigmas * wm;
    let y = sigmas - na::SMatrix::<f64, S, 9>::from_columns(&[x, x, x, x, x, x, x, x, x]);
    let mut tmp = na::SMatrix::<f64, S, S>::zeros();
    for i in 0..9 {
        tmp += wc[i] * y.column(i) * y.column(i).transpose();
    }
    let p = tmp + cov;
    (x, p)
}

// 状態遷移関数
fn fx(mut x: na::Vector4<f64>, u: na::Vector1<f64>) -> na::Vector4<f64> {
    x[3] +=
        ((M1 + M2 + J1 / (R_W * R_W)) / D * M2 * G * L * x[2] - M2 * L / D / R_W * KT * u[0]) * DT;
    x[2] += x[3] * DT;
    x[1] += (-M2 * M2 * G * L * L / D * x[2] + (M2 * L * L + J2) / D / R_W * KT * u[0]) * DT;
    x[0] += x[1] * DT;
    x
}

// 観測関数
fn hx(x_act: na::Vector4<f64>) -> na::Vector2<f64> {
    vector![
        x_act[1], // 駆動輪のオドメトリ
        x_act[3], // 角速度
    ]
}

fn predict(
    x: &mut na::Vector4<f64>,
    u: na::Vector1<f64>,
    p: &mut na::Matrix4<f64>,
) -> na::SMatrix<f64, 4, 9> {
    let mut sigmas = sigma_points(x, p);
    for i in 0..9 {
        sigmas.set_column(i, &fx(sigmas.column(i).into_owned(), u));
    }
    let (wm, wc) = sigma_weight();
    (*x, *p) = unscented_transform(&sigmas, &wm, &wc, &Q);
    sigmas
}

// センサ出力をシミュレーション
fn sensor(x_act: na::Vector4<f64>, rng: &mut rand::rngs::ThreadRng) -> na::Vector2<f64> {
    let mut x_obs = vector![
        x_act[1], // 駆動輪のオドメトリ
        x_act[3], // 角速度
    ];
    for i in 0..2 {
        let normal = Normal::new(0.0, R[(i, i)]).unwrap();
        x_obs[i] += normal.sample(rng);
    }
    x_obs
}

fn update(
    x_odom: &mut na::Vector4<f64>,
    x_obs: na::Vector2<f64>,
    p: &mut na::Matrix4<f64>,
    sigmas_f: &na::SMatrix<f64, 4, 9>,
) {
    let mut sigmas_h = na::SMatrix::<f64, 2, 9>::zeros();
    for i in 0..9 {
        sigmas_h.set_column(i, &hx(sigmas_f.column(i).into_owned()));
    }
    let (wm, wc) = sigma_weight();
    let (zp, pz) = unscented_transform(&sigmas_h, &wm, &wc, &R);
    let mut pxz = na::SMatrix::<f64, 4, 2>::zeros();
    for i in 0..9 {
        pxz += wc[i] * (sigmas_f.column(i) - *x_odom) * (sigmas_h.column(i) - zp).transpose();
    }
    let k = pxz * pz.try_inverse().expect("Inverse fail");
    *x_odom += k * (x_obs - zp);
    *p -= k * pz * k.transpose();
    // 対称性の維持
    *p = (*p + p.transpose()) / 2.0;
}

// 逐次実行版
fn dynamics(x: &mut na::Vector4<f64>, u: f64) {
    x[3] += ((M1 + M2 + J1 / (R_W * R_W)) / D * M2 * G * L * x[2] - M2 * L / D / R_W * KT * u) * DT;
    x[2] += x[3] * DT;
    x[1] += (-M2 * M2 * G * L * L / D * x[2] + (M2 * L * L + J2) / D / R_W * KT * u) * DT;
    x[0] += x[1] * DT;
}

#[allow(dead_code)]
fn dynamics_cpy(mut x: na::Vector4<f64>, u: f64) -> na::Vector4<f64> {
    dynamics(&mut x, u);
    x
}

// 参照軌道を生成する
// - 目標地点との偏差を取得
// - 偏差を上限速度でクリップして, 次の参照軌道とする
// - 参照軌道を上限でクリップする
// - 速度に応じて位置を動かす
// - 速度に応じて角度を決定する
fn next_plan(prev_plan: &na::Vector4<f64>) -> na::Vector4<f64> {
    let max_dx = [0.5 * DT, 1.2 * DT, 1.5 * DT, 5.0 * DT];
    let mut plan = *prev_plan;

    // x
    let d_x = 0.0 - prev_plan[0];
    let d_x = d_x.clamp(-max_dx[0], max_dx[0]);
    plan[0] += d_x;
    // plan[0] = plan[0].clamp(current[0] - 1.0, current[0] + 1.0);
    // x'
    let dd_x = d_x - prev_plan[1];
    plan[1] += dd_x.clamp(-max_dx[1], max_dx[1]);
    // plan[1] = plan[1].clamp(current[1] - 5.0, current[1] + 5.0);
    // plan[1] = plan[1].clamp(-3.0, 3.0);
    // θ
    let d_theta = d_x * 0.5 - prev_plan[2];
    plan[2] += d_theta.clamp(-max_dx[2], max_dx[2]);
    // θ'
    let dd_theta = d_theta * 3.0 - prev_plan[3];
    plan[3] += dd_theta.clamp(-max_dx[3], max_dx[3]);
    plan
}

// 参照経路(x_ref)を生成する
fn planning_err(x: &na::Vector4<f64>, x_plan: &na::Vector4<f64>) -> [f64; 4] {
    let x_g = x[0] + x[2] * L;
    let x_g_err = x_plan[0] - x_g;
    let x_g_dot = x[1] + x[3] * L;
    let x_g_dot_err = x_plan[1] - x_g_dot;
    let theta_err = x_plan[2] - x[2];
    let theta_dot_err = x_plan[3] - x[3];

    [x_g_err, x_g_dot_err, theta_err, theta_dot_err]
}

const GAIN: [f64; 5] = [0.5, 0.5, 16.0, 3.0, 0.1];
// const GAIN: [f64; 5] = [0.2, 0.2, 16.0, 1.0, 0.1];
// const GAIN: [f64; 5] = [0.2e-6, 0.2e-6, 16e-6, 1e-6, 0.1e-6];

// 評価関数は以下の通り
// J = \sum (x - x_ref)^2 + (θ-θ_ref)^2 + u^2
fn cost(mut x: na::Vector4<f64>, u: &[f64]) -> f64 {
    let mut c = 0.0;
    let mut plan = next_plan(&x);
    for e in u.iter() {
        dynamics(&mut x, *e);
        plan = next_plan(&plan);
        let x_err = planning_err(&x, &plan);
        // 追従誤差のコスト
        c += GAIN[0] * x_err[0].powi(2)
            + GAIN[1] * x_err[1].powi(4)
            + GAIN[2] * x_err[2].powi(4)
            + GAIN[3] * x_err[3].powi(4)
            + GAIN[4] * e.powi(2);
        c += 1e-6 * (x[2].cosh() - 1.2).max(0.0);
    }
    c
}

// コスト関数の勾配を数値微分によって求める(並列化版)
// 1. 数値微分の基礎となるコストを計算
// 2. 初期値xに運動方程式を適用し、kステップ先の状態を求める
// 3. kステップ先の状態と、入力に対して微小な変化を加えてコストを計算
// 4. 3.で求めたコストと1.のコストの差分を取り、微分を求める
fn grad_cost(x: &na::Vector4<f64>, u: &[f64; N], grad: &mut [f64]) {
    const EPS: f64 = 0.001;
    (0..N)
        .map(|i| dynamics_cpy(*x, u[i]))
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
    let file_path = "logs/op-mpc-x/op-mpc-x.csv";
    let mut wtr = csv::Writer::from_path(file_path)?;

    let tolerance = 1e-6;
    let lbfgs_memory = 20;
    let max_iters = usize::MAX;
    // let max_dur = std::time::Duration::from_millis(500);
    let max_dur = std::time::Duration::from_secs_f64(1.5);
    // let max_dur = std::time::Duration::from_secs_f64(DT);
    let mut panoc_cache = PANOCCache::new(N, tolerance, lbfgs_memory);

    let mut x_act = vector![0.5, 0.0, -0.15, 0.0];
    let mut u = [0.0; N];

    // UKFの初期化
    let mut rng = rand::thread_rng();
    let mut x_est = x_act;
    // let mut x_est = vector![2.0, 0.0, -0.2, 0.0];
    let mut p = matrix![
        10.0, 0.0, 0.0, 0.0;
        0.0, 10.0, 0.0, 0.0;
        0.0, 0.0, 10.0, 0.0;
        0.0, 0.0, 0.0, 10.0;
    ];

    let mut u_lpf = 0.0;

    const MAX_ITERS: usize = (10.0 / DT) as usize;
    for i in 0..MAX_ITERS + 1 {
        let f = |u: &[f64], c: &mut f64| -> Result<(), SolverError> {
            *c = cost(x_est, u);
            Ok(())
        };

        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            let u: [f64; N] = u.try_into().expect("slice with incorrect length");
            grad_cost(&x_est, &u, grad);
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

        let mut x_pred = x_est;
        for e in u.iter() {
            dynamics(&mut x_pred, *e);
        }
        let mut x_ref = x_est;
        for _ in 0..N {
            x_ref = next_plan(&x_ref);
        }

        u_lpf += (u[0] - u_lpf) * 0.5;
        u[0] = u_lpf;
        dynamics(&mut x_act, u[0]);

        // UKFの更新
        let sigmas_f = predict(&mut x_est, vector![u[0]], &mut p);
        let x_obs = sensor(x_act, &mut rng);
        update(&mut x_est, x_obs, &mut p, &sigmas_f);

        // let t = i as f64 * DT;
        // print!("{t:5.3}, {i:4}/{MAX_ITERS}, {:4}, ", status.iterations(),);
        print!("{:7.2}, ", u[0]);
        print!(
            "act: ({:7.2},{:7.2},{:7.2},{:7.2}) ",
            x_act[0], x_act[1], x_act[2], x_act[3]
        );
        print!(
            "est: ({:7.2},{:7.2},{:7.2},{:7.2}) ",
            x_est[0], x_est[1], x_est[2], x_est[3]
        );
        print!(
            "ref: ({:7.2},{:7.2},{:7.2},{:7.2}) ",
            x_ref[0], x_ref[1], x_ref[2], x_ref[3]
        );
        // print!(
        //     "p: ({:7.2},{:7.2},{:7.2},{:7.2})",
        //     p[(0, 0)],
        //     p[(1, 1)],
        //     p[(2, 2)],
        //     p[(3, 3)]
        // );
        println!();

        wtr.write_record(&[
            (i as f64 * DT).to_string(),
            u[0].to_string(),
            x_act[0].to_string(),
            x_act[1].to_string(),
            x_act[2].to_string(),
            x_act[3].to_string(),
            x_est[0].to_string(),
            x_est[1].to_string(),
            x_est[2].to_string(),
            x_est[3].to_string(),
            x_pred[0].to_string(),
            x_pred[1].to_string(),
            x_pred[2].to_string(),
            x_pred[3].to_string(),
            x_ref[0].to_string(),
            x_ref[1].to_string(),
            x_ref[2].to_string(),
            x_ref[3].to_string(),
        ])?;
        wtr.flush()?;

        // x[2]の絶対値がpi/2を超えればエラー
        if x_act[2].abs() > std::f64::consts::PI / 2.0 {
            bail!("Error: x[2] = {} > PI / 2", x_act[2])
        }
    }

    wtr.flush()?;
    Ok(())
}
