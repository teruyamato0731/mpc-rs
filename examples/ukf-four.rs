extern crate nalgebra as na;

use na::{matrix, vector};
use rand_distr::{Distribution, Normal};

const DT: f64 = 0.1;
const Q: na::Matrix4<f64> = matrix![
    1.0, 0.0, 0.0, 0.0;
    0.0, 1.0, 0.0, 0.0;
    0.0, 0.0, 1.0, 0.0;
    0.0, 0.0, 0.0, 1.0;
];
const R: na::Matrix2<f64> = matrix![
    0.25, 0.25;
    0.25, 0.25;
];

// LAMBDA = α^2 * (n + κ) - n = α^2 κ - n(1 - α^2)
const N: f64 = 4.0;
const ALPHA: f64 = 1e-3;
const BETA: f64 = 2.0;
const KAPPA: f64 = 3.0 - N;
const C: f64 = ALPHA * ALPHA * (N + KAPPA); // C := N + LAMBDA
const LAMBDA: f64 = C - N;

fn sigma_weight() -> (na::SVector<f64, 9>, na::SVector<f64, 9>) {
    let mut wm = na::SVector::<f64, 9>::from_element(1.0 / (2.0 * C));
    let mut wc = na::SVector::<f64, 9>::from_element(1.0 / (2.0 * C));
    wm[0] = LAMBDA / C;
    wc[0] = LAMBDA / C + 1.0 - ALPHA.powi(2) + BETA;
    (wm, wc)
}

fn sigma_points(x: &na::Vector4<f64>, p: &na::Matrix4<f64>) -> na::SMatrix<f64, 4, 9> {
    let l = (C * p).cholesky().unwrap().unpack();
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
    for i in 0..wc.ncols() {
        tmp += wc[i] * y.column(i) * y.column(i).transpose();
    }
    let p = tmp + cov;
    (x, p)
}

// 状態遷移関数
fn fx(mut x: na::Vector4<f64>, u: na::Vector2<f64>) -> na::Vector4<f64> {
    x[0] += x[1] * DT;
    x[1] += u[0] * DT;
    x[2] += x[3] * DT;
    x[3] += u[1] * DT;
    x
}

// 観測関数
fn hx(x: na::Vector4<f64>) -> na::Vector2<f64> {
    vector![x[0], x[2]]
}

fn predict(
    x: &mut na::Vector4<f64>,
    u: na::Vector2<f64>,
    p: &mut na::Matrix4<f64>,
) -> na::SMatrix<f64, 4, 9> {
    let mut sigmas = sigma_points(x, p);
    for i in 0..sigmas.ncols() {
        sigmas.set_column(i, &fx(sigmas.column(i).into_owned(), u));
    }
    let (wm, wc) = sigma_weight();
    (*x, *p) = unscented_transform(&sigmas, &wm, &wc, &Q);
    sigmas
}

// センサ出力をシミュレーション 位置を計測する
fn sensor(x_act: na::Vector4<f64>) -> na::Vector2<f64> {
    let normal = Normal::new(0.0, R[0]).unwrap();
    vector![
        x_act[0] + normal.sample(&mut rand::thread_rng()),
        x_act[2] + normal.sample(&mut rand::thread_rng()),
    ]
}

fn update(
    x_odom: &mut na::Vector4<f64>,
    x_obs: na::Vector2<f64>,
    p: &mut na::Matrix4<f64>,
    sigmas_f: &na::SMatrix<f64, 4, 9>,
) {
    let mut sigmas_h = na::SMatrix::<f64, 2, 9>::zeros();
    for i in 0..sigmas_h.ncols() {
        sigmas_h.set_column(i, &hx(sigmas_f.column(i).into_owned()));
    }
    let (wm, wc) = sigma_weight();
    let (zp, pz) = unscented_transform(&sigmas_h, &wm, &wc, &R);
    let mut pxz = na::SMatrix::<f64, 4, 2>::zeros();
    for i in 0..sigmas_f.ncols() {
        pxz += wc[i] * (sigmas_f.column(i) - *x_odom) * (sigmas_h.column(i) - zp).transpose();
    }
    let k = pxz * pz.try_inverse().unwrap();
    *x_odom += k * (x_obs - zp);
    *p -= k * pz * k.transpose();
}

fn main() {
    let mut x_act = vector![0.0, 0.0, 0.0, 0.0];
    let mut x_est = vector![0.0, 0.0, 0.0, 0.0];
    let mut p = matrix![
       10.0, 0.0, 0.0, 0.0;
       0.0, 10.0, 0.0, 0.0;
       0.0, 0.0, 10.0, 0.0;
       0.0, 0.0, 0.0, 10.0;
    ];
    for _ in 0..100 {
        let u = vector![0.5, -0.5];
        x_act = fx(x_act, u);
        let sigmas_f = predict(&mut x_est, u, &mut p);
        let x_obs = sensor(x_act);
        update(&mut x_est, x_obs, &mut p, &sigmas_f);

        print!(
            "x_act: ({:7.2},{:7.2},{:7.2},{:7.2}) ",
            x_act[0], x_act[1], x_act[2], x_act[3]
        );
        print!("x_obs: ({:7.2},{:7.2}) ", x_obs[0], x_obs[1]);
        println!(
            "x_est: ({:7.2},{:7.2},{:7.2},{:7.2}) ",
            x_est[0], x_est[1], x_est[2], x_est[3]
        );
        // println!("p: {}", p);
    }
}
