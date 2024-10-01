#[derive(Debug, Clone, Copy)]
pub struct Gaussian {
    pub mean: f64,
    pub var: f64,
}

impl Gaussian {
    pub fn new(mean: f64, var: f64) -> Self {
        Self { mean, var }
    }
}

impl Default for Gaussian {
    fn default() -> Self {
        Self {
            mean: 0.0,
            var: 0.0,
        }
    }
}

impl core::ops::Add for Gaussian {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            mean: self.mean + rhs.mean,
            var: self.var + rhs.var,
        }
    }
}

impl core::ops::Sub for Gaussian {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            mean: self.mean - rhs.mean,
            var: self.var - rhs.var,
        }
    }
}

impl core::ops::Mul for Gaussian {
    type Output = Gaussian;

    fn mul(self, rhs: Gaussian) -> Gaussian {
        let mean = (self.var * rhs.mean + rhs.var * self.mean) / (self.var + rhs.var);
        let var = (self.var * rhs.var) / (self.var + rhs.var);
        Gaussian { mean, var }
    }
}

impl core::ops::Mul<f64> for Gaussian {
    type Output = Gaussian;

    fn mul(self, rhs: f64) -> Gaussian {
        Gaussian {
            mean: self.mean * rhs,
            var: self.var * rhs,
        }
    }
}
