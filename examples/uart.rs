use mpc::packet::{Control, State};
use std::io::{BufRead, BufReader};
use std::time::Duration;

fn main() {
    let mut port = serialport::new("/dev/ttyUSB0", 115_200)
        .timeout(Duration::from_millis(10))
        .open()
        .expect("Failed to open port");

    let mut reader = BufReader::new(port.try_clone().unwrap());

    loop {
        let c = Control { u: 1234 };
        let cobs = c.as_cobs();
        port.write_all(&cobs).expect("Write failed!");

        // データが読み込まれるまで待機
        let mut buf = Vec::new();
        if let Ok(len) = reader.read_until(0x00, &mut buf) {
            println!("{:?}", buf);
            // 18バイト読み込まれたら処理を行う
            if len == 18 {
                if let Ok(array) = buf.try_into() {
                    let s = State::from_cobs(&array).unwrap();
                    println!("{:?}", s);
                } else {
                    eprintln!("Failed to convert buffer to array");
                }
            }
        }
    }
}
