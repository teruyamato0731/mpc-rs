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

#[derive(Debug, AsBytes, FromBytes, FromZeroes)]
#[repr(C)]
pub struct Sensor {
    pub encoder: [i16; 2], // 駆動輪のオドメトリ
    pub gyro: f32,         // ジャイロセンサの角速度
}

#[derive(Debug, AsBytes, FromBytes, FromZeroes)]
#[repr(C)]
pub struct Sensor2 {
    pub encoder: [i16; 2], // 駆動輪のオドメトリ
    pub gyro: f32,         // ジャイロセンサの角速度
    pub accel: [f32; 2],   // 加速度センサ
}

#[derive(Debug, AsBytes, FromBytes, FromZeroes)]
#[repr(packed)]
pub struct Sensor3 {
    pub enable: u8,
    pub encoder: [i16; 2], // 駆動輪のオドメトリ
    pub gyro: f32,         // ジャイロセンサの角速度
    pub accel: [f32; 2],   // 加速度センサ
}

macro_rules! impl_cobs_convertible {
    ($type:ty) => {
        impl $type {
            pub const SIZE: usize = std::mem::size_of::<Self>();
            pub const BUF_SIZE: usize = Self::SIZE + 2;

            // cobsバイト列に変換
            pub fn as_cobs(&self) -> [u8; Self::BUF_SIZE] {
                let buf: [u8; Self::SIZE] = self.as_bytes().try_into().unwrap();
                cobs_rs::stuff(buf, 0)
            }
            // cobsバイト列から復元
            pub fn from_cobs(buf: &[u8; Self::BUF_SIZE]) -> Option<Self> {
                let (cobs, _): ([u8; Self::SIZE], _) = cobs_rs::unstuff(*buf, 0);
                Self::read_from(&cobs)
            }
        }
    };
}

impl_cobs_convertible!(State);
impl_cobs_convertible!(Control);
impl_cobs_convertible!(Sensor);
impl_cobs_convertible!(Sensor2);
impl_cobs_convertible!(Sensor3);

impl Control {
    pub const MAX: i16 = 10000;
    pub fn from_current(current: f64) -> Self {
        const K: f64 = Control::MAX as f64 / 10.0;
        let u = (K * current) as i16;
        Control { u }
    }
}

impl From<State> for na::Vector4<f64> {
    fn from(s: State) -> Self {
        na::Vector4::new(s.x as f64, s.dx as f64, s.theta as f64, s.dtheta as f64)
    }
}

impl From<Sensor> for na::Vector3<f64> {
    fn from(s: Sensor) -> Self {
        na::Vector3::new(s.encoder[0] as f64, s.encoder[1] as f64, s.gyro as f64)
    }
}

impl From<Sensor2> for na::Vector5<f64> {
    fn from(s: Sensor2) -> Self {
        na::Vector5::new(
            s.encoder[0] as f64,
            s.encoder[1] as f64,
            s.gyro as f64,
            s.accel[0] as f64,
            s.accel[1] as f64,
        )
    }
}

impl Sensor3 {
    pub fn parse(s: Sensor3) -> (u8, na::Vector5<f64>) {
        let mut result = na::Vector5::new(
            s.encoder[0] as f64,
            s.encoder[1] as f64,
            s.gyro as f64,
            s.accel[0] as f64,
            s.accel[1] as f64,
        );

        // s.enable の bit が 0 なら 0 にする
        for i in 0..5 {
            if (s.enable & (1 << i)) == 0 {
                result[i] = 0.0;
            }
        }

        (s.enable, result)
    }
}
