use rand::prelude::*;
use rand::rngs::ThreadRng;
use rayon::prelude::*;

// MPPI (Model Predictive Path Integral) controller
pub struct Mppi<const N: usize, const K: usize> {
    rng: ThreadRng,
    dist: rand_distr::Normal<f64>,
    limit: (f64, f64),
    dynamics: fn(&na::Vector4<f64>, f64) -> na::Vector4<f64>,
    cost: fn(&na::Vector4<f64>) -> f64,
    lambda: f64,
    std_dev: f64,
}

impl<const N: usize, const K: usize> Mppi<N, K> {
    pub fn new(
        dynamics: fn(&na::Vector4<f64>, f64) -> na::Vector4<f64>,
        cost: fn(&na::Vector4<f64>) -> f64,
        lambda: f64,
        std_dev: f64,
        limit: (f64, f64),
    ) -> Self {
        let rng = rand::thread_rng();
        let dist = rand_distr::Normal::<f64>::new(0.0, std_dev).unwrap();
        Self {
            rng,
            dist,
            limit,
            dynamics,
            cost,
            lambda,
            std_dev,
        }
    }

    // MPPIの実装
    pub fn compute(
        &mut self,
        x: &na::Vector4<f64>,
        u_n: &na::SVector<f64, N>,
    ) -> Result<na::SVector<f64, N>, &'static str> {
        let v_k_n: Vec<na::SVector<f64, N>> = (0..K)
            .map(|_| {
                u_n + na::SVector::<f64, N>::from_fn(|_, _| {
                    self.dist
                        .sample(&mut self.rng)
                        .clamp(self.limit.0, self.limit.1)
                })
            })
            .collect();

        // 並列処理でコスト・重みの計算を行う
        let mut c_k = vec![0.0; K];
        let mut w_k = vec![0.0; K];
        let sum: f64 = v_k_n
            .par_iter()
            .zip(c_k.par_iter_mut())
            .zip(w_k.par_iter_mut())
            .map(|((v_k, c_i), w_i)| {
                // コストの計算
                let (cost, _) = v_k.iter().fold((0.0, *x), |(c, x_c), v| {
                    // 状態の更新
                    let x_n = (self.dynamics)(&x_c, *v);
                    // コストの累積
                    (c + (self.cost)(&x_n), x_n)
                });
                *c_i = cost;
                // 重みの計算
                let cost_term = cost / self.lambda;
                let control_term = u_n
                    .iter()
                    .zip(v_k.iter())
                    .fold(0.0, |acc, (u, v)| acc + u / self.std_dev * v);
                *w_i = (-cost_term - control_term).exp();
                *w_i
            })
            .reduce(|| 0.0, |a, b| a + b);
        // 正規化項
        if sum == 0.0 {
            return Err("sum is zero");
        }
        // 重み付け平均
        let u_n = w_k
            .par_iter()
            .enumerate()
            .map(|(i, w)| *w * v_k_n[i] / sum)
            .reduce(na::SVector::<f64, N>::zeros, |acc, x| acc + x);

        // u が 不正値の場合は終了
        if u_n[0].is_nan() || u_n[0].is_infinite() {
            return Err("u is invalid");
        }

        Ok(u_n)
    }
}
