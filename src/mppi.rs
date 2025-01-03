use rand::prelude::*;
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;

// MPPI (Model Predictive Path Integral) controller
#[derive(Clone)]
pub struct Mppi<const N: usize, const K: usize, const S: usize> {
    limit: (f64, f64),
    dynamics: fn(&na::SVector<f64, S>, f64) -> na::SVector<f64, S>,
    cost: fn(&na::SVector<f64, S>) -> f64,
    lambda: f64,
    std_dev: f64,
}

impl<const N: usize, const K: usize, const S: usize> Mppi<N, K, S> {
    pub fn new(
        dynamics: fn(&na::SVector<f64, S>, f64) -> na::SVector<f64, S>,
        cost: fn(&na::SVector<f64, S>) -> f64,
        lambda: f64,
        std_dev: f64,
        limit: (f64, f64),
    ) -> Self {
        Self {
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
        x: &na::SVector<f64, S>,
        u_n: &na::SVector<f64, N>,
    ) -> Result<na::SVector<f64, N>, &'static str> {
        let dist = rand_distr::Normal::<f64>::new(0.0, self.std_dev).unwrap();
        let v_k_n = (0..K)
            .into_par_iter()
            .map_init(Xoshiro256Plus::from_entropy, |rng, _| {
                (u_n + na::SVector::<f64, N>::from_distribution(&dist, rng))
                    .map(|v| v.clamp(self.limit.0, self.limit.1))
            })
            .collect::<Vec<_>>();

        // 並列処理でコスト・重みの計算を行う
        let inv = self.std_dev.powi(-2);
        let c_k = v_k_n
            .par_iter()
            .map(|v_k| {
                // コストの計算
                let (cost, _) = v_k.iter().fold((0.0, *x), |(c, x_c), v| {
                    // 状態の更新
                    let x_n = (self.dynamics)(&x_c, *v);
                    // コストの累積
                    (c + (self.cost)(&x_n), x_n)
                });
                // 重みの計算
                let control_term: f64 = u_n.iter().zip(v_k.iter()).map(|(u, v)| u * inv * v).sum();
                -cost - control_term
            })
            .collect::<Vec<_>>();
        // コストの平行移動
        let max = *c_k
            .par_iter()
            .filter(|&&c| c.is_finite())
            .max_by(|&a, &b| a.partial_cmp(b).unwrap())
            .ok_or("Cannot calculate max")?;
        // Softmax関数 in-place で計算
        let mut w_k = c_k;
        w_k.par_iter_mut()
            .for_each(|c| *c = ((*c - max) / self.lambda).exp());
        let sum: f64 = w_k.par_iter().sum();
        // 正規化項
        if sum == 0.0 {
            return Err("sum is zero");
        }
        // 重み付け平均
        let u_n = w_k
            .into_par_iter()
            .zip(v_k_n.into_par_iter())
            .map(|(w, v_k)| w / sum * v_k)
            .sum::<na::SVector<f64, N>>();

        // u が 不正値の場合は終了
        if u_n[0].is_nan() || u_n[0].is_infinite() {
            return Err("u is invalid");
        }

        Ok(u_n)
    }
}
