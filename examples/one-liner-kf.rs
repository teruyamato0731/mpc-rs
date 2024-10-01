use mpc::gaussian::Gaussian;
use rand_distr::{Distribution, Normal};

const PROCESS_STDDEV: f64 = 1.0;
const PROCESS_VAR: f64 = PROCESS_STDDEV * PROCESS_STDDEV;
const SENSOR_STDDEV: f64 = 2.0;
const SENSOR_VAR: f64 = SENSOR_STDDEV * SENSOR_STDDEV;

fn dynamics(x: f64, dx: f64) -> f64 {
    x + dx
}

fn predict(mut x: Gaussian, u: Gaussian) -> Gaussian {
    x.mean = dynamics(x.mean, u.mean);
    x.var += u.var;
    x
}

// センサ出力をシミュレーション
fn sensor(x_act: f64) -> Gaussian {
    let normal = Normal::new(x_act, SENSOR_STDDEV).unwrap();
    let noise = normal.sample(&mut rand::thread_rng());
    Gaussian::new(noise, SENSOR_VAR)
}

fn update(x_odom: Gaussian, x_obs: Gaussian) -> Gaussian {
    x_odom * x_obs
}

fn _update(x_odom: Gaussian, x_obs: Gaussian) -> Gaussian {
    // カルマンゲイン (予測の信用割合)
    let k_gain = x_odom.var / (x_odom.var + x_obs.var);
    // 残差
    let y = x_obs.mean - x_odom.mean;
    // 事後分布の平均
    let x_est_mean = x_odom.mean + k_gain * y;
    // 事後分布の分散
    let x_est_var = (1.0 - k_gain) * x_odom.var;
    Gaussian::new(x_est_mean, x_est_var)
}

fn _predict(mut x: Gaussian, u: Gaussian) -> Gaussian {
    x.mean = dynamics(x.mean, u.mean);
    x.var += u.var;
    x
}

fn main() {
    let mut x_act = 0.0;
    let mut x_est = Gaussian::new(10.0, 10.0);
    // ↑誤った初期予測を与える
    for _ in 0..100 {
        let u = 0.5;
        x_act = dynamics(x_act, u);
        x_est = predict(x_est, Gaussian::new(u, PROCESS_VAR));
        let x_obs = sensor(x_act);
        x_est = update(x_est, x_obs);

        print!("x_act: {:6.2}, ", x_act);
        print!("x_obs: {:6.2}, ", x_obs.mean);
        print!("x_est.mean: {:6.2}, ", x_est.mean);
        println!("x_est.var: {:7.3}", x_est.var);
    }
}
