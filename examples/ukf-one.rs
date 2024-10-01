extern crate nalgebra as na;

use rand_distr::{Distribution, Normal};

const DT: f64 = 1.0;
const Q: f64 = 1.0;
const R: f64 = 1.0;

// n: 次元数
// α: シグマ点のばらつき (0 < α < 1)
// β: ガウス分布ならば 2.0
// κ: 3 - n とする
// λ = α^2 * (n + κ) - n
const N: f64 = 1.0;
const ALPHA: f64 = 1e-3;
const BETA: f64 = 2.0;
const KAPPA: f64 = 3.0 - N;
const C: f64 = ALPHA * ALPHA * (N + KAPPA); // C := N + LAMBDA
const LAMBDA: f64 = ALPHA * ALPHA * (N + KAPPA) - N;

fn sigma_weight() -> ([f64; 3], [f64; 3]) {
    let wm = [LAMBDA / C, 1.0 / (2.0 * C), 1.0 / (2.0 * C)];
    let wc = [
        LAMBDA / C + 1.0 - ALPHA.powi(2) + BETA,
        1.0 / (2.0 * C),
        1.0 / (2.0 * C),
    ];
    (wm, wc)
}

fn sigma_points(x: f64, p: f64) -> [f64; 3] {
    let s = C * p;
    let u = s.sqrt();
    [x, x + u, x - u]
}

fn unscented_transform(sigmas: &[f64; 3], wm: &[f64; 3], wc: &[f64; 3], cov: f64) -> (f64, f64) {
    let mut x = 0.0;
    for i in 0..3 {
        x += wm[i] * sigmas[i];
    }
    let mut p = 0.0;
    for i in 0..3 {
        p += wc[i] * (sigmas[i] - x).powi(2);
    }
    p += cov;
    (x, p)
}

// 状態遷移関数
fn fx(x: f64, u: f64) -> f64 {
    x + u * DT
}

// 観測関数
fn hx(x: f64) -> f64 {
    x
}

fn predict(x: &mut f64, u: f64, p: &mut f64) -> [f64; 3] {
    let mut sigmas = sigma_points(*x, *p);
    for p in sigmas.iter_mut() {
        *p = fx(*p, u);
    }
    let (wm, wc) = sigma_weight();
    (*x, *p) = unscented_transform(&sigmas, &wm, &wc, Q);
    sigmas
}

// センサ出力をシミュレーション 位置を計測する
fn sensor(x_act: f64) -> f64 {
    let normal = Normal::new(x_act, R).unwrap();
    normal.sample(&mut rand::thread_rng())
}

fn update(x_odom: &mut f64, x_obs: f64, p: &mut f64, sigmas_f: &[f64; 3]) {
    let mut sigmas_h = [0.0; 3];
    for i in 0..3 {
        sigmas_h[i] = hx(sigmas_f[i]);
    }
    let (wm, wc) = sigma_weight();
    let (zp, pz) = unscented_transform(&sigmas_h, &wm, &wc, R);
    let mut pxz = 0.0;
    for i in 0..3 {
        pxz += wc[i] * (sigmas_f[i] - *x_odom) * (sigmas_h[i] - zp);
    }
    let k = pxz / pz;
    *x_odom += k * (x_obs - zp);
    *p -= k * pz * k;
    // println!("x_odom: {:6.3} p: {:6.3}", x_odom, p);
}

fn main() {
    let mut x_act = 0.0;
    let mut x_est = 10.0;
    let mut p = 100.0;
    for _ in 0..100 {
        let u = 0.5;
        x_act = fx(x_act, u);
        let sigmas_f = predict(&mut x_est, u, &mut p);
        let x_obs = sensor(x_act);
        update(&mut x_est, x_obs, &mut p, &sigmas_f);

        print!("x_act: {:6.3} ", x_act);
        print!("x_obs: {:6.3} ", x_obs);
        print!("x_est: {:6.3} ", x_est);
        println!("p: {:6.3}", p);
    }

    let (wm, wc) = sigma_weight();
    println!("wm: {:?}", wm);
    println!("wc: {:?}", wc);
}
