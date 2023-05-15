use numrust::*;

fn main() {
    let x = vec![1., 2., 3., 4., 5.];
    let y = vec![1., 3., 2.0, 5.0, 4.0];
    println!("cov of x and y is {:?}", covariance(&x, &y));
    println!("cov of x and y is {:?}", corrcoef(&x, &y));
}
