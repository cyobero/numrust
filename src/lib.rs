use either::Either;
use rand::prelude::*;
use rand_distr::{Binomial, Normal, Uniform, WeightedIndex};
use std::collections::HashMap;

use rand::Rng;
/// Selects a random sample from a given population with optional replacement and/or
/// custom probabilities for each element.
///
/// # Arguments
///
/// * `a`: The population to sample from.
/// * `size`: An optional parameter specifying the size of the sample. If `None`, a single
///   element will be sampled. If `Some(n)`, `n` elements will be sampled.
/// * `replace`: A boolean indicating whether or not sampling should be done with replacement.
///   If `true`, elements may be sampled more than once. If `false`, each element may only be
///   sampled once.
/// * `p`: An optional slice of floats representing the probabilities of each element being
///   sampled. The length of this slice must be the same as the length of the population.
///   If `None`, uniform probabilities will be used.
///
/// # Returns
///
/// A vector of sampled elements from the population.
///
/// # Examples
///
/// ```
/// use numrust::choice;
/// let population = vec![1, 2, 3, 4, 5];
///
/// // Sample 3 elements without replacement.
/// let sample = choice(&population, Some(3), false, None);
///
/// // Sample a single element with replacement using custom probabilities.
/// let p = [0.1, 0.1, 0.2, 0.3, 0.3];
/// let sample = choice(&population, None, true, Some(&p));
/// ```
///
/// # Panics
///
/// This function will panic if `size` is larger than the length of the population and `replace`
/// is set to `false`, or if `p` is provided and its length does not match the length of the
/// population.
///
pub fn choice<T: Clone>(a: &[T], size: Option<usize>, replace: bool, p: Option<&[f64]>) -> Vec<T> {
    let mut rng = rand::thread_rng();

    let population_size = a.len();
    let sample_size = size.unwrap_or(1);

    let mut probabilities = HashMap::new();
    if let Some(p_vals) = p {
        for (i, &val) in p_vals.iter().enumerate() {
            probabilities.insert(i, val);
        }
    } else {
        for i in 0..population_size {
            probabilities.insert(i, 1.0 / population_size as f64);
        }
    }

    // Create a distribution to randomly select indices from a population,
    // using a uniform distribution if 'replace' is true, or a weighted
    // distribution if 'replace' is false.
    let distribution: Either<Uniform<usize>, WeightedIndex<f64>> = if replace {
        Either::Left(Uniform::new(0, population_size))
    } else {
        let indices: Vec<usize> = (0..population_size).collect();
        let weights: Vec<f64> = indices.iter().map(|i| probabilities[i]).collect();
        Either::Right(WeightedIndex::new(weights).unwrap())
    };

    let mut sample = Vec::with_capacity(sample_size);
    for _ in 0..sample_size {
        let random_index = match &distribution {
            Either::Left(d) => d.sample(&mut rng),
            Either::Right(d) => d.sample(&mut rng),
        };
        sample.push(a[random_index].clone());
        if !replace {
            let p_val = probabilities[&random_index];
            let remaining_mass = 1.0 - p_val;
            let remaining_indices: Vec<usize> = probabilities
                .iter()
                .filter(|&(i, _)| *i != random_index)
                .map(|(i, _)| *i)
                .collect();
            for index in remaining_indices {
                probabilities.insert(index, probabilities[&index] / remaining_mass);
            }
        }
    }

    sample
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
/// use numrust::binomial;
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

        let data = binomial(n, p, size);

        let actual_mean = data.iter().sum::<u64>() as f64 / size as f64;
        let actual_std = ((data
            .iter()
            .map(|&x| (x as f64 - actual_mean).powi(2))
            .sum::<f64>()
            / size as f64)
            .sqrt())
        .round();

        let expected_mean = n as f64 * p;
        let expected_std = ((n as f64 * p * (1.0 - p)).sqrt()).round();

        assert_abs_diff_eq!(actual_mean, expected_mean, epsilon = 0.1);
        assert_eq!(actual_std, expected_std, "actual_std: {}", actual_std);
    }

    #[test]
    fn test_single_choice() {
        let a = vec![1, 2, 3, 4, 5];
        let sample = choice(&a, Some(1), true, None);
        assert_eq!(sample.len(), 1);
        assert!(a.contains(&sample[0]));
    }

    #[test]
    fn test_multiple_choices() {
        let a = vec![1, 2, 3, 4, 5];
        let sample = choice(&a, Some(3), true, None);
        assert_eq!(sample.len(), 3);
        assert!(sample.iter().all(|x| a.contains(x)));
    }

    #[test]
    fn test_choice_without_replacement() {
        let a = vec![1, 2, 3, 4, 5];
        let sample = choice(&a, Some(3), false, None);
        assert_eq!(sample.len(), 3);
        assert!(sample.iter().all(|x| a.contains(x)));
    }
}
