use numrust::*;

fn main() {
    let range = [1, 2, 3, 4, 5];
    let x = choice(
        &["red", "blue", "green"],
        Some(3),
        false,
        Some(&[0.1, 0.3, 0.3]),
    );

    println!("{:?}", x);
}
