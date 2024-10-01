extern crate nalgebra as na;

use na::{matrix, vector};
use rand_distr::{Distribution, Normal};

const DT: f64 = 0.1;
const Q: na::Matrix2<f64> = matrix![0.25, 0.5; 0.5, 1.0];
const R: na::Matrix1<f64> = matrix![2.0];

// LAMBDA = α^2 * (n + κ) - n = α^2 κ - n(1 - α^2)
const N: f64 = 2.0;
const ALPHA: f64 = 1e-3;
const BETA: f64 = 2.0;
const KAPPA: f64 = 3.0 - N;
const C: f64 = ALPHA * ALPHA * (N + KAPPA); // C := N + LAMBDA
const LAMBDA: f64 = C - N;

fn sigma_weight() -> (na::Vector5<f64>, na::Vector5<f64>) {
    let wm = vector![
        LAMBDA / C,
        1.0 / (2.0 * C),
        1.0 / (2.0 * C),
        1.0 / (2.0 * C),
        1.0 / (2.0 * C),
    ];
    let wc = vector![
        LAMBDA / C + 1.0 - ALPHA.powi(2) + BETA,
        1.0 / (2.0 * C),
        1.0 / (2.0 * C),
        1.0 / (2.0 * C),
        1.0 / (2.0 * C),
    ];
    (wm, wc)
}

fn sigma_points(x: &na::Vector2<f64>, p: &na::Matrix2<f64>) -> na::Matrix2x5<f64> {
    let l = (C * p).cholesky().unwrap().unpack();
    na::Matrix2x5::from_columns(&[
        *x,
        *x + l.column(0),
        *x - l.column(0),
        *x + l.column(1),
        *x - l.column(1),
    ])
}

fn unscented_transform<const S: usize>(
    sigmas: &na::SMatrix<f64, S, 5>,
    wm: &na::Vector5<f64>,
    wc: &na::Vector5<f64>,
    cov: &na::SMatrix<f64, S, S>,
) -> (na::SVector<f64, S>, na::SMatrix<f64, S, S>) {
    let x = sigmas * wm;
    let y = sigmas - na::SMatrix::<f64, S, 5>::from_columns(&[x, x, x, x, x]);
    let mut tmp = na::SMatrix::<f64, S, S>::zeros();
    for i in 0..5 {
        tmp += wc[i] * y.column(i) * y.column(i).transpose();
    }
    let p = tmp + cov;
    (x, p)
}

// 状態遷移関数
fn fx(mut x: na::Vector2<f64>, u: na::Vector2<f64>) -> na::Vector2<f64> {
    x[0] += x[1].powi(4) * DT;
    x[1] += (u[0] - u[1]) * DT;
    x
}

// 観測関数
fn hx(x: na::Vector2<f64>) -> na::Vector1<f64> {
    vector![x[0]]
}

fn predict(
    x: &mut na::Vector2<f64>,
    u: na::Vector2<f64>,
    p: &mut na::Matrix2<f64>,
) -> na::Matrix2x5<f64> {
    let mut sigmas = sigma_points(x, p);
    for i in 0..5 {
        sigmas.set_column(i, &fx(sigmas.column(i).into_owned(), u));
    }
    let (wm, wc) = sigma_weight();
    (*x, *p) = unscented_transform(&sigmas, &wm, &wc, &Q);
    sigmas
}

// センサ出力をシミュレーション 位置を計測する
fn sensor(x_act: na::Vector2<f64>) -> na::Vector1<f64> {
    let normal = Normal::new(x_act[0], R[0]).unwrap();
    vector![normal.sample(&mut rand::thread_rng())]
}

fn update(
    x_odom: &mut na::Vector2<f64>,
    x_obs: na::Vector1<f64>,
    p: &mut na::Matrix2<f64>,
    sigmas_f: &na::Matrix2x5<f64>,
) {
    let mut sigmas_h = na::Matrix1x5::<f64>::zeros();
    for i in 0..5 {
        sigmas_h.set_column(i, &hx(sigmas_f.column(i).into_owned()));
    }
    let (wm, wc) = sigma_weight();
    let (zp, pz) = unscented_transform(&sigmas_h, &wm, &wc, &R);
    let mut pxz = na::SMatrix::<f64, 2, 1>::zeros();
    for i in 0..5 {
        pxz += wc[i] * (sigmas_f.column(i) - *x_odom) * (sigmas_h.column(i) - zp);
    }
    let k = pxz * pz.try_inverse().unwrap();
    *x_odom += k * (x_obs - zp);
    *p -= k * pz * k.transpose();
}

fn main() {
    let (wm, wc) = sigma_weight();
    println!("wm: {:?}", wm);
    println!("wc: {:?}", wc);

    let mut x_act = vector![0.0, 0.0];
    let mut x_est = vector![0.0, 0.0];
    let mut p = matrix![10.0, 0.0; 0.0, 10.0];
    for _ in 0..100 {
        let u = vector![0.5, -0.5];
        x_act = fx(x_act, u);
        let sigmas_f = predict(&mut x_est, u, &mut p);
        let x_obs = sensor(x_act);
        update(&mut x_est, x_obs, &mut p, &sigmas_f);

        print!("x_act: ({:7.2},{:7.2}) ", x_act[0], x_act[1]);
        print!("x_obs: {:7.2}, ", x_obs[0]);
        print!("x_est: ({:7.2},{:7.2}), ", x_est[0], x_est[1]);
        println!("p: {:?}", p);
    }
}
