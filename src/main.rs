use numrust::*;

fn main() {
    let x = vec![1., 2., 3.];
    let y = vec![4., 8., 6.];
    println!("cov of x and y is {:?}", covariance(&x, &y));
    println!("cov of x and y is {:?}", corrcoef(&x, &y));
}
