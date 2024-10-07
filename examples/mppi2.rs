extern crate nalgebra as na;
use rand::prelude::*;
use rayon::prelude::*;

// ref: https://zenn.dev/takuya_fukatsu/articles/36c0d6911d18b7
// cargo run --example mppi2 --release

// 予測ホライゾン
const T: f32 = 2.0;
const N: usize = 40;
const DT: f32 = T / N as f32;

// 制御ホライゾン
const K: usize = 8000;
const LAMBDA: f32 = 2.5;
const R: f32 = 1.0;

// 制約
const LIMIT: (f32, f32) = (-3.0, 3.0);

// 系ダイナミクスを記述
fn dynamics(state: &na::Vector2<f32>, control: f32) -> na::Vector2<f32> {
    let mut next = *state;
    next[0] += state[1] * DT;
    next[1] += control * DT;
    next
}

fn main() {
    let mut x = na::Vector2::new(1.0, 0.0);
    let mut u_n = na::SVector::<f32, N>::zeros();

    let mut rng = rand::thread_rng();
    let dist = rand_distr::Normal::<f32>::new(u_n[0], R).unwrap();

    // 5s までシミュレーション
    let mut t = 0.0;
    while t < 5.0 {
        let v_k_n: Vec<na::SVector<f32, N>> = (0..K)
            .map(|_| {
                na::SVector::<f32, N>::from_fn(|_, _| dist.sample(&mut rng).clamp(LIMIT.0, LIMIT.1))
            })
            .collect();

        // 並列処理で予測とコスト計算を行う
        let mut c_k = [0.0; K];
        c_k.par_iter_mut().enumerate().for_each(|(i, c_i)| {
            // コストの計算
            let (cost, _) = v_k_n[i].iter().fold((0.0, x), |(c, x_c), v| {
                // 状態の更新
                let x_n = dynamics(&x_c, *v);
                // コストの累積
                (c + x_n[0].powi(2) + x_n[1].powi(2), x_n)
            });
            *c_i = cost;
        });

        // 重みの計算
        let mut w_k = [0.0; K];
        w_k.par_iter_mut().enumerate().for_each(|(i, w_i)| {
            let cost_term = c_k[i] / LAMBDA;
            let control_term = u_n
                .iter()
                .zip(v_k_n[i].iter())
                .fold(0.0, |acc, (u, v)| acc + u / R * v);
            *w_i = (-cost_term - control_term).exp();
        });
        // 正規化
        let sum: f32 = w_k.iter().sum();
        w_k.iter_mut().for_each(|w| *w /= sum);

        // 重み付け平均
        u_n = w_k
            .par_iter()
            .enumerate()
            .map(|(i, w)| *w * v_k_n[i])
            .reduce(na::SVector::<f32, N>::zeros, |acc, x| acc + x);

        x = dynamics(&x, u_n[0]);

        println!(
            "t: {:.2}, u: {:5.2}, x: [{:.2}, {:.2}]",
            t, u_n[0], x[0], x[1]
        );

        // u が 不正値の場合は終了
        if u_n[0].is_nan() || u_n[0].is_infinite() {
            break;
        }

        t += DT;
    }
}
