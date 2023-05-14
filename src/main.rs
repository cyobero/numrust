use numrust::*;

fn main() {
    let mean = 10.0;
    let std = 2.0;
    let n = 10000;

    let data = normal(mean, std, n);

    let actual_mean = data.iter().sum::<f64>() / n as f64;
    let actual_std =
        (data.iter().map(|x| (x - actual_mean).powi(2)).sum::<f64>() / n as f64).sqrt();

    println!("mean: {}, actual_mean: {}", mean, actual_mean);
    println!("std: {}, actual_std: {}", std, actual_std);
}
