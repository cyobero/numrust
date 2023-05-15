use numrust::*;

fn main() {
    let colors = vec!["red", "blue", "green"];
    let p = vec![0.7, 0.2, 0.1];

    let mut reds = 0;
    let mut blues = 0;
    let mut greens = 0;

    let sample = choice(&colors, 2, false, None);
    for color in sample {
        match color {
            "red" => reds += 1,
            "blue" => blues += 1,
            "green" => greens += 1,
            _ => (),
        }
    }

    println!("red: {}", reds);
    println!("blue: {}", blues);
    println!("green: {}", greens);
}
