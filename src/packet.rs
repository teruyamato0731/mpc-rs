use zerocopy::{AsBytes, FromBytes, FromZeroes};

// 状態変数 x, \dot{x}, \theta, \theta
#[derive(Debug, AsBytes, FromBytes, FromZeroes)]
#[repr(C)]
pub struct State {
    pub x: f32,
    pub dx: f32,
    pub theta: f32,
    pub dtheta: f32,
}

#[derive(Debug, AsBytes, FromBytes, FromZeroes)]
#[repr(C)]
pub struct Control {
    pub u: i16,
}

impl State {
    // cobsバイト列に変換
    pub fn as_cobs(&self) -> [u8; 18] {
        let buf: [u8; 16] = self.as_bytes().try_into().unwrap();
        cobs_rs::stuff(buf, 0)
    }
    // cobsバイト列から復元
    pub fn from_cobs(buf: &[u8; 18]) -> Option<Self> {
        let (cobs, _): ([u8; 16], _) = cobs_rs::unstuff(*buf, 0);
        Self::read_from(&cobs)
    }
}

impl Control {
    // cobsバイト列に変換
    pub fn as_cobs(&self) -> [u8; 4] {
        let buf: [u8; 2] = self.as_bytes().try_into().unwrap();
        cobs_rs::stuff(buf, 0)
    }
    // cobsバイト列から復元
    pub fn from_cobs(buf: &[u8; 4]) -> Option<Self> {
        let (cobs, _): ([u8; 2], _) = cobs_rs::unstuff(*buf, 0);
        Self::read_from(&cobs)
    }
}
