use rand::prelude::*;
use rand_distr::{Binomial, Normal};

use rand::Rng;

pub fn binomial(n: u64, p: f64, size: usize) -> Vec<u64> {
    let mut rng = thread_rng();
    let binom = Binomial::new(n, p).unwrap();
    let mut nums = Vec::with_capacity(size);
    for _ in 0..size {
        let num = binom.sample(&mut rng);
        nums.push(num);
    }
    nums
}

/// Generates a vector of `n` random samples from a normal (Gaussian) distribution
/// with the specified `mean` and `standard deviation`.
///
/// # Arguments
///
/// * `mean` - the mean of the normal distribution
/// * `std` - the standard deviation of the normal distribution
/// * `n` - the number of samples to generate
///
/// # Returns
///
/// A `Vec` of `n` random samples from a normal distribution with the specified mean
/// and standard deviation.
///
/// # Example
///
/// ```
/// use numrust::normal;
/// use rand_distr::Normal;
/// use rand::prelude::*;
///
/// fn main() {
///     let mean = 0.0;
///     let std = 1.0;
///     let n = 100;
///
///     let data = normal(mean, std, n);
///
///     println!("{:?}", data);
/// }
/// ```
///
/// This example generates 100 random samples from a standard normal distribution (i.e.
/// a normal distribution with mean 0 and standard deviation 1) and prints them to the console.
///
/// # Panics
///
/// This function will panic if the `Normal::new` constructor fails to create a normal distribution
/// with the specified mean and standard deviation.
pub fn normal(mean: f64, std: f64, n: usize) -> Vec<f64> {
    let normal = Normal::new(mean, std).unwrap();
    let mut rng = thread_rng();
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let sample = normal.sample(&mut rng);
        data.push(sample);
    }

    data
}

/// Calculates the mean value of a slice of f64 values.
///
/// # Arguments
///
/// * `nums` - A slice of f64 values
///
/// # Example
///
/// ```
/// let nums = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let mean = numrust::mean(&nums);
/// assert_eq!(mean, 3.0);
/// ```
///
/// # Panics
///
/// The `mean` function does not panic.
pub fn mean(nums: &[f64]) -> f64 {
    let sum: f64 = nums.iter().sum();
    let mean = sum / (nums.len() as f64);
    mean
}

/// Calculates the variance of a slice of f64 values.
///
/// # Arguments
///
/// * `nums` - A slice of f64 values
///
/// # Returns
///
/// The variance of the input slice, or NaN if the slice is empty.
///
/// # Example
///
/// ```
/// let nums = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let var = numrust::variance(&nums);
/// assert_eq!(var, 2.5);
/// ```
///
/// # Panics
///
/// The `variance` function does not panic.
pub fn variance(nums: &[f64]) -> f64 {
    let mean = mean(nums);
    if nums.len() == 0 {
        f64::NAN
    } else {
        nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ((nums.len() - 1) as f64)
    }
}

/// Calculates the standard deviation of a slice of f64 values.
///
/// # Arguments
///
/// * `nums` - A slice of f64 values
///
/// # Returns
///
/// The standard deviation of the input slice, or NaN if the slice is empty.
///
/// # Example
///
/// ```
/// let nums = [1.0, 2.0, 3.0];
/// let std_dev = numrust::std_dev(&nums);
/// assert_eq!(std_dev, 1.0);
/// ```
///
/// # Panics
///
/// The `std_dev` function does not panic.
pub fn std_dev(nums: &[f64]) -> f64 {
    let var = variance(nums);
    if nums.len() == 0 {
        f64::NAN
    } else {
        var.sqrt()
    }
}

/// Generate a list of `n` random integers between `min` (inclusive) and `max` (exclusive).
///
/// # Examples
///
/// ```
/// use numrust::randint;
///
/// let values = randint(0, 10, 5);
/// assert_eq!(values.len(), 5);
/// assert!(values.iter().all(|&x| x >= 0 && x < 10));
/// ```
///
/// # Panics
///
/// This function will panic if `min >= max`.
///
pub fn randint(min: i32, max: i32, n: usize) -> Vec<i32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(min..max)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mean_empty() {
        let nums: [f64; 0] = [];
        assert!(mean(&nums).is_nan());
    }

    #[test]
    fn test_mean_single() {
        let nums = [42.0];
        assert_eq!(mean(&nums), 42.0);
    }

    #[test]
    fn test_mean_multiple() {
        let nums = [1.0, 2.0, 3.0];
        assert_eq!(mean(&nums), 2.0);
    }

    #[test]
    fn test_variance_empty() {
        let nums: [f64; 0] = [];
        assert!(variance(&nums).is_nan());
    }

    #[test]
    fn test_variance_single() {
        let nums = [42.0];
        assert!(variance(&nums).is_nan());
    }

    #[test]
    fn test_variance_multiple() {
        let nums = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(variance(&nums), 2.5);
    }

    #[test]
    fn test_std_dev_empty() {
        let nums: [f64; 0] = [];
        assert!(std_dev(&nums).is_nan());
    }

    #[test]
    fn test_std_dev_single() {
        let nums = [42.0];
        assert!(std_dev(&nums).is_nan());
    }

    #[test]
    fn test_std_dev_multiple() {
        let nums = [1.0, 2.0, 3.0];
        assert_eq!(std_dev(&nums), 1.0);
    }

    #[test]
    fn test_randint_within_range() {
        let min = 1;
        let max = 10;
        let n = 100;
        let values = randint(min, max, n);
        assert!(values.iter().all(|&x| x >= min && x < max));
    }

    #[test]
    fn test_randint_correct_length() {
        let min = 1;
        let max = 10;
        let n = 100;
        let values = randint(min, max, n);
        assert_eq!(values.len(), n);
    }

    #[test]
    #[should_panic]
    fn test_randint_panics_on_invalid_input() {
        let min = 10;
        let max = 1;
        let n = 100;
        randint(min, max, n);
    }

    #[test]
    fn test_normal_returns_correct_number_of_samples() {
        let mean = 0.0;
        let std = 1.0;
        let n = 100;

        let data = normal(mean, std, n);

        assert_eq!(data.len(), n);
    }

    #[test]
    fn test_normal_returns_samples_with_correct_mean_and_std() {
        let mean = 10.0;
        let std = 2.0;
        let n = 10000;

        let data = normal(mean, std, n);

        let actual_mean = data.iter().sum::<f64>() / n as f64;
        let actual_std = ((data.iter().map(|x| (x - actual_mean).powi(2)).sum::<f64>() / n as f64)
            .sqrt())
        .round();

        assert_abs_diff_eq!(actual_mean, mean, epsilon = 0.1);
        assert_eq!(actual_std, std, "actual_std: {}", actual_std);
    }
}
