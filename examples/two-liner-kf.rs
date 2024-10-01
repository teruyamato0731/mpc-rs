extern crate nalgebra as na;
use na::{matrix, vector};
use rand_distr::{Distribution, Normal};

const DT: f64 = 0.01;
const R: na::Matrix1<f64> = matrix![4.0];
const F: na::Matrix2<f64> = matrix![1.0, DT; 0.0, 1.0];
// ref: https://inzkyk.xyz/kalman_filter/kalman_filter_math/#section:7.3
const Q: na::Matrix2<f64> = matrix![0.25, 0.5; 0.5, 1.0];
const H: na::Matrix1x2<f64> = matrix![1.0, 0.0];
const B: na::Matrix2<f64> = matrix![0.0, 0.0; 1.0, -1.0];

fn dynamics(x: na::Vector2<f64>, u: na::Vector2<f64>) -> na::Vector2<f64> {
    F * x + B * u
}

fn predict(
    mut x: na::Vector2<f64>,
    u: na::Vector2<f64>,
    p: &mut na::Matrix2<f64>,
) -> na::Vector2<f64> {
    x = dynamics(x, u);
    *p = F * *p * F.transpose() + Q;
    x
}

// センサ出力をシミュレーション 位置を計測する
fn sensor(x_act: na::Vector2<f64>) -> na::Vector1<f64> {
    let normal = Normal::new(x_act[0], R[0]).unwrap();
    vector![normal.sample(&mut rand::thread_rng())]
}

fn update(
    x_odom: na::Vector2<f64>,
    x_obs: na::Vector1<f64>,
    p: &mut na::Matrix2<f64>,
) -> na::Vector2<f64> {
    // カルマンゲイン (予測の信用割合)
    let s = H * *p * H.transpose() + R;
    let k = *p * H.transpose() * s.try_inverse().unwrap();
    // let k = *p * H.transpose() * s.pseudo_inverse(1e-30).unwrap();
    // 残差
    let y = x_obs - H * x_odom;
    // 事後分布の平均
    let x_est = x_odom + k * y;
    // 事後分布の分散
    // *p = *p - k * H * *p;
    // 上式よりも下式のほうが数値的に安定 ref: https://inzkyk.xyz/kalman_filter/kalman_filter_math/#section:7.4
    *p = (na::Matrix2::identity() - k * H) * *p * (na::Matrix2::identity() - k * H).transpose()
        + k * R * k.transpose();
    x_est
}

fn main() {
    let mut x_act = vector![0.0, 0.0];
    let mut x_est = vector![0.0, 0.0];
    let mut p = matrix![100.0, 0.0;
                                                              0.0, 100.0];
    for _ in 0..100 {
        let u = vector![0.5, -0.5];
        x_act = dynamics(x_act, u);
        x_est = predict(x_est, u, &mut p);
        let x_obs = sensor(x_act);
        x_est = update(x_est, x_obs, &mut p);

        print!("x_act: ({:6.2},{:6.2}) ", x_act[0], x_act[1]);
        print!("x_obs: {:6.2}, ", x_obs[0]);
        print!("x_est: ({:6.2},{:6.2}), ", x_est[0], x_est[1]);
        println!("p: {:?}", p);
    }
}
