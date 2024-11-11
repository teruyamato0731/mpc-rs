// 状態空間の次元
const N: usize = 6;
// 観測空間の次元
const O: usize = 5;
// シグマ点の数
const M: usize = 2 * N + 1;

type State = na::SVector<f64, N>;
type Cov<const N: usize> = na::SMatrix<f64, N, N>;
type Sigma<const S: usize> = na::SMatrix<f64, S, M>;

pub struct UnscentedKalmanFilter {
    x: State,
    p: Cov<N>,
    q: Cov<N>,
    r: Cov<O>,
    wm: Sigma<1>,
    wc: Sigma<1>,
    sigma_f: Sigma<N>,
}

impl UnscentedKalmanFilter {
    const N: f64 = N as f64;
    const ALPHA: f64 = 1e-3;
    const BETA: f64 = 2.0;
    const KAPPA: f64 = 3.0 - Self::N;
    const C: f64 = Self::ALPHA * Self::ALPHA * (Self::N + Self::KAPPA);
    const LAMBDA: f64 = Self::C - Self::N;

    pub fn new(x: State, p: Cov<N>, q: Cov<N>, r: Cov<O>) -> Self {
        let (wm, wc) = Self::sigma_weight();
        let sigma_f = Sigma::<N>::from_element(f64::NAN);
        Self {
            x,
            p,
            q,
            r,
            wm,
            wc,
            sigma_f,
        }
    }

    pub fn predict<F>(&mut self, u: f64, fx: F)
    where
        F: Fn(&State, f64) -> State,
    {
        self.sigma_f = Self::compute_sigma_points(&self.x, &self.p, u, fx);
        let (x, p) = Self::unscented_transform(&self.sigma_f, &self.wm, &self.wc, &self.q);
        self.x = x;
        self.p = p;
    }

    pub fn update<F>(&mut self, x_obs: &na::SVector<f64, O>, hx: F)
    where
        F: Fn(&State) -> na::SVector<f64, O>,
    {
        let mut sigmas_h = na::SMatrix::<f64, O, M>::zeros();
        for i in 0..M {
            sigmas_h.set_column(i, &hx(&self.sigma_f.column(i).into_owned()));
        }
        let (zp, pz) = Self::unscented_transform(&sigmas_h, &self.wm, &self.wc, &self.r);
        let mut pxz = na::SMatrix::<f64, N, O>::zeros();
        for i in 0..M {
            pxz += self.wc[i]
                * (self.sigma_f.column(i) - self.x)
                * (sigmas_h.column(i) - zp).transpose();
        }
        let k = pxz * pz.try_inverse().expect("Inverse fail");
        self.x += k * (x_obs - zp);
        self.p -= k * pz * k.transpose();
        // 対称性の維持
        self.p = (self.p + self.p.transpose()) / 2.0;
    }

    fn compute_sigma_points<F>(x: &State, p: &Cov<N>, u: f64, fx: F) -> Sigma<N>
    where
        F: Fn(&State, f64) -> State,
    {
        let mut sigma_f = Self::sigma_points(x, p);
        for i in 0..M {
            sigma_f.set_column(i, &fx(&sigma_f.column(i).into_owned(), u));
        }
        sigma_f
    }

    // 推定した状態を返す
    pub fn state(&self) -> State {
        self.x
    }

    pub fn covariance(&self) -> Cov<N> {
        self.p
    }

    pub fn set_q(&mut self, q: Cov<N>) {
        self.q = q;
    }

    fn unscented_transform<const S: usize>(
        sigmas: &Sigma<S>,
        wm: &Sigma<1>,
        wc: &Sigma<1>,
        cov: &Cov<S>,
    ) -> (na::SVector<f64, S>, Cov<S>) {
        let x = sigmas * wm.transpose();
        let y = sigmas - Sigma::<S>::from_columns(&[x; M]);
        let mut tmp = Cov::<S>::zeros();
        for i in 0..M {
            tmp += wc[i] * y.column(i) * y.column(i).transpose();
        }
        let p = tmp + cov;
        (x, p)
    }

    fn sigma_weight() -> (Sigma<1>, Sigma<1>) {
        let mut wm = Sigma::<1>::from_element(1.0 / (2.0 * Self::C));
        let mut wc = Sigma::<1>::from_element(1.0 / (2.0 * Self::C));
        wm[0] = Self::LAMBDA / Self::C;
        wc[0] = Self::LAMBDA / Self::C + 1.0 - Self::ALPHA.powi(2) + Self::BETA;
        (wm, wc)
    }

    fn sigma_points(x: &State, p: &Cov<N>) -> Sigma<N> {
        let svd = (Self::C * p).svd_unordered(true, false);
        let s_sqrt = Cov::<N>::from_diagonal(&svd.singular_values.map(|x| x.sqrt()));
        let u = svd.u.unwrap();
        let l = u * s_sqrt;
        let mut sigma_points = Sigma::<N>::zeros();
        sigma_points.set_column(0, x);
        for i in 0..N {
            sigma_points.set_column(1 + i, &(x + l.column(i)));
            sigma_points.set_column(1 + N + i, &(x - l.column(i)));
        }
        sigma_points
    }
}
