extern crate nalgebra as na;
use rand::prelude::*;
use rayon::prelude::*;

// ref: https://zenn.dev/teruyamato0731/scraps/bc2b2b7c96bd07
// cargo run --example mppi4 --release

// 予測ホライゾン
const T: f64 = 2.0;
const N: usize = 20;
const DT: f64 = T / N as f64;

// 制御ホライゾン
const K: usize = 1e5 as usize;
const LAMBDA: f64 = 2.5;
const R: f64 = 1.0;

// 制約
const LIMIT: (f64, f64) = (-20.0, 20.0);

// 系ダイナミクスを記述
const M1: f64 = 150e-3;
const R_W: f64 = 50e-3;
const M2: f64 = 2.3 - 2.0 * M1 + 2.0;
const L: f64 = 0.2474; // 重心までの距離
const J1: f64 = M1 * R_W * R_W;
const J2: f64 = 0.5;
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

fn cost(x: &na::Vector4<f64>) -> f64 {
    20.0 * x[0].powi(2) + 20.0 * x[1].powi(2) + 100.0 * x[2].powi(2) + 10.0 * x[3].powi(2)
}

fn main() {
    let mut x = na::Vector4::new(1.0, 0.0, 0.1, 0.0);
    let mut u_n = na::SVector::<f64, N>::zeros();

    let mut rng = rand::thread_rng();
    let dist = rand_distr::Normal::<f64>::new(u_n[0], R).unwrap();

    // 5s までシミュレーション
    let mut t = 0.0;
    while t < 5.0 {
        let v_k_n: Vec<na::SVector<f64, N>> = (0..K)
            .map(|_| {
                na::SVector::<f64, N>::from_fn(|_, _| dist.sample(&mut rng).clamp(LIMIT.0, LIMIT.1))
            })
            .collect();

        // 並列処理で予測とコスト計算を行う
        let mut c_k = vec![0.0; K];
        c_k.par_iter_mut().enumerate().for_each(|(i, c_i)| {
            // コストの計算
            let (cost, _) = v_k_n[i].iter().fold((0.0, x), |(c, x_c), v| {
                // 状態の更新
                let x_n = dynamics(&x_c, *v);
                // コストの累積
                (c + cost(&x_n), x_n)
            });
            *c_i = cost;
        });

        // println!("c_k: {:?}", c_k);

        // 重みの計算
        let mut w_k = vec![0.0; K];
        w_k.par_iter_mut().enumerate().for_each(|(i, w_i)| {
            let cost_term = c_k[i] / LAMBDA;
            let control_term = u_n
                .iter()
                .zip(v_k_n[i].iter())
                .fold(0.0, |acc, (u, v)| acc + u / R * v);
            *w_i = (-cost_term - control_term).exp();
        });
        // 正規化
        let sum: f64 = w_k.iter().sum();
        w_k.iter_mut().for_each(|w| *w /= sum);

        // 重み付け平均
        u_n = w_k
            .par_iter()
            .enumerate()
            .map(|(i, w)| *w * v_k_n[i])
            .reduce(na::SVector::<f64, N>::zeros, |acc, x| acc + x);

        x = dynamics(&x, u_n[0]);

        println!(
            "t: {:.2}, u: {:5.2}, x: [{:.2}, {:.2}, {:.2}, {:.2}]",
            t, u_n[0], x[0], x[1], x[2], x[3]
        );

        // u が 不正値の場合は終了
        if u_n[0].is_nan() || u_n[0].is_infinite() {
            break;
        }

        // x[2] が 60度 以上になったら終了
        if x[2].abs() > 60.0f64.to_radians() {
            break;
        }

        t += DT;
    }
}
