extern crate nalgebra as na;

pub mod gaussian;
pub mod ukf;

#[derive(Debug, Clone, Default)]
pub struct Dynamics {
    pub position: f32,
    pub velocity: f32,
}

impl Dynamics {
    pub fn update(&mut self, force: f32) {
        self.position += self.velocity;
        self.velocity += 0.1 + force;
    }
}

pub struct Mpc {
    pub dynamics: Dynamics,
}

impl Mpc {
    pub fn new(dynamics: Dynamics) -> Self {
        Self { dynamics }
    }
    pub fn update(&mut self, target: f32) -> f32 {
        let force = (target - self.dynamics.position) * 0.1;
        self.dynamics.update(force);
        force
    }
}
