use rand::prelude::*;
use rand::Rng;
use rand_distr::{Binomial, Normal, WeightedIndex};

/// Returns a vector of `size` elements randomly chosen from the array `a`.
///
/// # Arguments
///
/// * `a` - An array of elements to choose from.
/// * `size` - The number of elements to choose.
/// * `replace` - If `true`, elements are drawn with replacement. If `false`, elements are drawn without replacement.
/// * `p` - If `Some`, a slice of probabilities associated with each element in `a`. The length of `p` must be the same as the length of `a`.
///
/// # Returns
///
/// A vector of elements randomly chosen from `a`.
///
/// # Panics
///
/// * If `size` is greater than the length of `a` and `replace` is `false`.
/// * If `p` is `Some` and the length of `p` is not equal to the length of `a`.
///
/// # Examples
///
/// ```
/// use numrust::random::choice;
///
/// let colors = ["blue", "green", "red"];
///
/// // Draw 5 elements with replacement
/// let choices = choice(&colors, 5, true, None);
/// assert_eq!(choices.len(), 5);
/// ```
pub fn choice<T: Clone>(a: &[T], size: usize, replace: bool, p: Option<&[f64]>) -> Vec<T> {
    if !replace & (size > a.len()) {
        panic!("`size` cannot be greater than the length of `a` if `replace` is false");
    }

    if p.is_some() {
        if p.unwrap().len() != a.len() {
            panic!("`a` must be the same length as `p`");
        }
    }

    let mut rng = rand::thread_rng();
    let dist = match p {
        Some(probs) => WeightedIndex::new(probs).unwrap(),
        None => WeightedIndex::new(vec![1.0 / a.len() as f64; a.len()]).unwrap(),
    };
    let mut result = Vec::with_capacity(size);
    if replace {
        for _ in 0..size {
            result.push(a[dist.sample(&mut rng)].clone());
        }
    } else {
        let mut indices = (0..a.len()).collect::<Vec<_>>();
        for _ in 0..size {
            let i = dist.sample(&mut rng);
            result.push(a[indices[i]].clone());
            if !replace {
                indices.swap_remove(i);
            }
        }
    }
    result
}

/// Generates samples from a binomial distribution with parameters `n` and `p`.
///
/// The binomial distribution models the number of successes in a fixed number of independent
/// Bernoulli trials, where each trial has a probability `p` of success. The `n` parameter specifies
/// the number of trials, and `p` specifies the probability of success.
///
/// # Arguments
///
/// * `n` - The number of Bernoulli trials.
/// * `p` - The probability of success in each Bernoulli trial.
/// * `size` - The number of samples to generate.
///
/// # Panics
///
/// This function will panic if the `Binomial::new` constructor fails. This can occur if `n` is zero,
/// if `p` is not between 0 and 1, or if the `Binomial` distribution is not supported by the underlying
/// random number generator.
///
/// # Returns
///
/// A `Vec` containing `size` samples drawn from the binomial distribution with parameters `n` and `p`.
///
/// # Examples
///
/// ```
/// use numrust::random::binomial;
/// use rand_distr::Binomial;
/// use rand::thread_rng;
///
/// fn main() {
///     let n = 10;
///     let p = 0.5;
///     let size = 100;
///
///     let data = binomial(n, p, size);
///
///     assert_eq!(data.len(), size);
/// }
/// ```
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
/// use numrust::random::normal;
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

/// Generate a list of `n` random integers between `min` (inclusive) and `max` (exclusive).
///
/// # Examples
///
/// ```
/// use numrust::random::randint;
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
mod numrust_random_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
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

        let actual_mean = crate::mean(&data);
        let actual_std = crate::std_dev(&data).round();

        assert_abs_diff_eq!(actual_mean, mean, epsilon = 0.05);
        assert_eq!(actual_std, std, "actual_std: {}", actual_std);
    }

    #[test]
    fn test_binomial_returns_correct_number_of_samples() {
        let n = 10;
        let p = 0.5;
        let size = 100;

        let data = binomial(n, p, size);

        assert_eq!(data.len(), size);
    }

    #[test]
    fn test_binomial_returns_samples_between_zero_and_n() {
        let n = 10;
        let p = 0.5;
        let size = 100;

        let data = binomial(n, p, size);

        assert!(data.iter().all(|&x| x <= n));
        assert!(data.iter().all(|&x| x as f64 >= 0f64));
    }

    #[test]
    fn test_binomial_returns_samples_with_correct_mean_and_std() {
        let n = 10;
        let p = 0.7;
        let size = 10000;

        let data = binomial(n, p, size)
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<f64>>();

        let actual_mean = crate::mean(&data);
        let actual_std = crate::std_dev(&data).round();

        let expected_mean = n as f64 * p;
        let expected_std = ((n as f64 * p * (1.0 - p)).sqrt()).round();

        assert_abs_diff_eq!(actual_mean, expected_mean, epsilon = 0.05);
        assert_eq!(actual_std, expected_std, "actual_std: {}", actual_std);
    }

    #[test]
    fn test_choice_uniform() {
        let a = vec![1, 2, 3, 4, 5];
        let result = choice(&a, 10, true, None);
        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|x| a.contains(x)));
    }

    #[test]
    fn test_choice_probabilities() {
        let a = vec![1, 2, 3, 4, 5];
        let p = vec![0.1, 0.2, 0.3, 0.2, 0.2];
        let result = choice(&a, 10, true, Some(&p));
        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|x| a.contains(x)));

        let a = vec!["red", "blue", "green"];
        let sample = choice(&a, 10000, true, Some(&[0.7, 0.2, 0.05]));
        let mut reds = 0;
        let mut blues = 0;
        let mut greens = 0;
        for color in sample {
            match color {
                "red" => reds += 1,
                "blue" => blues += 1,
                "green" => greens += 1,
                _ => (),
            }
        }
        // The counts of each color should be reds > blues > greens.
        assert!((reds > blues) & (blues > greens));
    }

    #[test]
    #[should_panic(
        expected = "`size` cannot be greater than the length of `a` if `replace` is false"
    )]
    fn test_choice_size_mismatch() {
        let a = vec![1, 2, 3, 4, 5];
        choice(&a, 10, false, None);
    }

    #[test]
    #[should_panic(expected = "`a` must be the same length as `p`")]
    fn test_choice_probabilities_length_mismatch() {
        let a = vec![1, 2, 3, 4, 5];
        let p = vec![0.1, 0.2, 0.3, 0.2];
        choice(&a, 10, true, Some(&p));
    }
}
