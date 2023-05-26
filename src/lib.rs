pub mod random;
use std::error::Error;
use std::fmt;

#[derive(Debug, PartialEq)]
pub struct ArangeError(String);

impl fmt::Display for ArangeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: Step size cannot be 0.", self.0)
    }
}

impl Error for ArangeError {}

/// Represents a trait for computing statistical moments of an array.
pub trait Moment {
    /// Computes the mean (average) of the array.
    fn mean(&self) -> Option<f64>;

    /// Computes the variance of the array.
    fn var(&self) -> Option<f64>;

    /// Computes the standard deviation of the array.
    fn std(&self) -> Option<f64>;

    /// Computes the skewness of the array.
    fn skew(&self) -> Option<f64>;
}

impl<T: Into<f64> + Copy> Moment for [T] {
    fn mean(&self) -> Option<f64> {
        if self.len() == 0 {
            None
        } else {
            Some(mean(self))
        }
    }

    fn var(&self) -> Option<f64> {
        if self.len() == 0 {
            None
        } else {
            Some(variance(self))
        }
    }

    fn std(&self) -> Option<f64> {
        if self.len() == 0 {
            None
        } else {
            Some(std_dev(self))
        }
    }

    fn skew(&self) -> Option<f64> {
        if self.len() == 0 {
            None
        } else {
            Some(skew(self))
        }
    }
}

/// Computes the Pearson correlation coefficient between two arrays of floats.
///
/// # Arguments
///
/// * `x`: A slice of floats representing the first variable.
/// * `y`: A slice of floats representing the second variable.
///
/// # Returns
///
/// A 2x2 array of floats containing the correlation matrix, where the (0, 1) and (1, 0)
/// entries are the correlation coefficients between `x` and `y`.
///
/// # Examples
///
/// ```
/// use numrust::corrcoef;
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [1.0, 3.0, 2.0, 5.0, 4.0];
/// let corr = corrcoef(&x, &y);
/// let expected_result = [[1.0, 0.7999999999999998], [0.7999999999999998, 1.0]];
/// assert_eq!(corr, expected_result);
/// ```
///
/// # Panics
///
/// This function will panic if the input arrays `x` and `y` have different lengths.
pub fn corrcoef(x: &[f64], y: &[f64]) -> [[f64; 2]; 2] {
    if x.len() != y.len() {
        panic!("x and y must have the same length");
    }
    let cov = covariance(x, y)[0][1];
    let x_std = std_dev(&x);
    let y_std = std_dev(&y);
    let corr = cov / (x_std * y_std);
    [[1.0, corr], [corr, 1.0]]
}

/// Calculates the covariance matrix for two vectors of float values `x` and `y`.
///
/// # Arguments
///
/// * `x` - A slice of float values representing the first vector.
/// * `y` - A slice of float values representing the second vector.
///
/// # Returns
///
/// A 2x2 fixed-size array of float values representing the covariance matrix for `x` and `y`. The first
/// row and column of the matrix correspond to `x`, while the second row and column correspond to
/// `y`. The element at position `(i, j)` represents the covariance between the `i`th variable
/// (either `x` or `y`) and the `j`th variable.
///
/// # Examples
///
/// ```
/// use numrust::covariance;
/// use approx::assert_abs_diff_eq;
///
/// let x = &[1.0, 2.0, 3.0];
/// let y = &[4.0, 8.0, 6.0];
/// let cov = covariance(x, y);
///
/// // The covariance between x and x should be the variance of x
/// assert_eq!(cov[0][0], 1.0);
///
/// // The covariance between x and y should be the same as the covariance between y and x
/// assert_eq!(cov[0][1], cov[1][0]);
///
/// // The covariance between y and y should be the variance of y
/// assert_eq!(cov[1][1], 4.0);
/// ```
///
/// # Panics
///
/// Panics if `x` and `y` have different lengths.
///
pub fn covariance(x: &[f64], y: &[f64]) -> [[f64; 2]; 2] {
    if x.len() != y.len() {
        panic!("x and y must have the same length");
    }
    let n = x.len() as f64;
    let x_mean = mean(x);
    let y_mean = mean(y);
    let mut cov = [[0.0; 2]; 2];
    cov[0][0] = x.iter().map(|&a| (a - x_mean) * (a - x_mean)).sum::<f64>() / (n - 1.0);
    cov[0][1] = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - x_mean) * (b - y_mean))
        .sum::<f64>()
        / (n - 1.0);
    cov[1][0] = cov[0][1];
    cov[1][1] = y.iter().map(|&b| (b - y_mean) * (b - y_mean)).sum::<f64>() / (n - 1.0);
    cov
}
/// Generates a sequence of evenly spaced values within a specified range.
///
/// # Arguments
///
/// * `start` - The starting value of the sequence.
/// * `stop` - The end value of the sequence (exclusive).
/// * `step` - The step size between each value in the sequence. A positive value generates
///            increasing values, while a negative value generates decreasing values.
///
/// # Returns
///
/// A `Vec<f64>` containing the generated sequence of values.
///
/// # Examples
///
/// ```
/// use numrust::arange;
///
/// let sequence = arange(0, 5, 1).unwrap();
/// assert_eq!(sequence, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
///
/// let sequence = arange(0, 5, 2).unwrap();
/// assert_eq!(sequence, vec![0.0, 2.0, 4.0]);
///
/// let sequence = arange(5, 0, -1).unwrap();
/// assert_eq!(sequence, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
/// ```
///
/// # Panics
///
/// Panics if `step` is equal to zero.
pub fn arange(start: usize, stop: usize, step: isize) -> Result<Vec<f64>, ArangeError> {
    if step == 0 {
        Err(ArangeError("Step size cannot be 0".to_string()))
    } else {
        let result: Vec<f64> = if step > 0 {
            (start..stop)
                .step_by(step as usize)
                .map(|x| x as f64)
                .collect()
        } else {
            ((stop as isize - step) as usize..=start)
                .rev()
                .step_by((-step) as usize)
                .map(|x| x as f64)
                .collect()
        };
        Ok(result)
    }
}

/// Calculates the mean value of a slice of f64 values.
///
/// # Arguments
///
/// * `nums` - A slice of f64 values
///
/// # Returns
///
/// A f64 of the computed mean of `nums`
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
pub fn mean<T: Into<f64> + Copy>(nums: &[T]) -> f64 {
    let sum: f64 = nums.iter().map(|&x| x.into()).sum();
    let mean = sum / (nums.len() as f64);
    mean
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
pub fn std_dev<T: Into<f64> + Copy>(nums: &[T]) -> f64 {
    let var = variance(nums);
    if nums.len() == 0 {
        f64::NAN
    } else {
        var.sqrt()
    }
}

/// Calculates the sample variance of a slice of f64 values.
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
pub fn variance<T: Into<f64> + Copy>(nums: &[T]) -> f64 {
    let mean = mean(nums);
    if nums.len() == 0 {
        f64::NAN
    } else {
        nums.iter().map(|&x| (x.into() - mean).powi(2)).sum::<f64>() / ((nums.len() - 1) as f64)
    }
}
/// Calculates the skewness of a slice of numeric values.
///
/// # Arguments
///
/// * `nums` - A reference to a slice of values of any type that can be converted into `f64`.
///
/// # Returns
///
/// The calculated skewness value as an `f64`.
///
/// # Examples
///
/// ```
/// use numrust::skew;
/// use approx::assert_abs_diff_eq;
///
/// let nums = vec![6, 6, 6, 9];
/// assert_abs_diff_eq!(skew(&nums), 1.1547, epsilon = 0.001);
///
/// let nums = [-1.0, -0.5, 1.0, 2.5];
/// assert_abs_diff_eq!(skew(&nums), 0.3651, epsilon = 0.001);
///
/// let nums = [1, 15];
/// assert_eq!(skew(&nums), 0.0);
///
/// let nums = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(skew(&nums), 0.0);
/// ```
///
/// # Panics
///
/// This function will panic if `nums` is an empty slice.
pub fn skew<T: Into<f64> + Copy>(nums: &[T]) -> f64 {
    let mean = mean(nums);
    let variance = nums.iter().map(|&x| (x.into() - mean).powi(2)).sum::<f64>() / nums.len() as f64;
    let std_dev = variance.sqrt();
    let skewness = nums
        .iter()
        .map(|&x| (x.into() - mean) / std_dev)
        .map(|x| x.powi(3))
        .sum::<f64>()
        / nums.len() as f64;
    skewness
}

#[cfg(test)]
mod numrust_tests {
    use std::assert_eq;

    use approx::assert_abs_diff_eq;

    use super::*;
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
    fn test_arange() {
        let start = 1;
        let stop = 10;
        let step = 3;

        let expected_result = vec![1.0, 4.0, 7.0];
        assert_eq!(arange(start, stop, step).unwrap(), expected_result);

        let start = 5;
        let stop = 20;
        let step = 5;

        let expected_result = vec![5.0, 10.0, 15.0];
        assert_eq!(arange(start, stop, step).unwrap(), expected_result);

        assert_eq!(arange(1, 6, 1).unwrap(), vec![1., 2., 3., 4., 5.]);
    }

    #[test]
    fn test_arange_with_zero_step() {
        let start = 0;
        let stop = 10;
        let step = 0;

        assert!(arange(start, stop, step).is_err());
    }

    #[test]
    fn test_arange_with_negative_step() {
        let start = 10;
        let stop = 0;
        let step = -2;

        let expected_result = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert_eq!(arange(start, stop, step).unwrap(), expected_result);

        let seq = arange(5, 0, -1).unwrap();
        assert_eq!(seq, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_covariance() {
        // test case 1
        let x = vec![1., 2., 3.];
        let y = vec![4., 8., 6.];
        assert_eq!(covariance(&x, &y)[0][0], 1.0);
        assert_eq!(covariance(&x, &y)[0][1], 1.0);
        assert_eq!(covariance(&x, &y)[1][0], 1.0);
        assert_eq!(covariance(&x, &y)[1][1], 4.);

        // test case 2
        let x = arange(0, 10, 2).unwrap();
        let y = arange(10, 0, -2).unwrap();
        assert_eq!(covariance(&x, &y)[0][0], 10.0);
        assert_eq!(covariance(&x, &y)[0][1], -10.0);
        assert_eq!(covariance(&x, &y)[1][0], -10.0);
        assert_eq!(covariance(&x, &y)[1][1], 10.0);

        // test case 3
        let x = vec![-1., -2., -3., -4., -5.];
        let y = vec![1., 3., 3., 4., 6.];
        assert_eq!(covariance(&x, &y)[0][0], 2.5);
        assert_eq!(covariance(&x, &y)[0][1], -2.75);
        assert_eq!(covariance(&x, &y)[1][0], -2.75);
        assert_abs_diff_eq!(covariance(&x, &y)[1][1], 3.3, epsilon = 0.00001);
    }

    #[test]
    #[should_panic(expected = "x and y must have the same length")]
    fn test_covariance_unequal_lengths() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0];
        covariance(&x, &y);
    }

    #[test]
    fn test_corrcoef() {
        let x = vec![1., 2., 3.];
        let y = vec![4., 5., 6.];
        let expected_result = [[1.0, 1.0], [1.0, 1.0]];
        assert_eq!(corrcoef(&x, &y), expected_result);

        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 3.0, 2.0, 5.0, 4.0];
        let corr = corrcoef(&x, &y);
        assert_abs_diff_eq!(corr[0][1], 0.8, epsilon = 0.000001);
        assert_abs_diff_eq!(corr[1][0], 0.8, epsilon = 0.000001);
    }

    #[test]
    #[should_panic(expected = "x and y must have the same length")]
    fn test_corrcoef_panics_on_unequal_length_inputs() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [0.5, 1.5, 2.5, 3.5];
        corrcoef(&x, &y);
    }

    #[test]
    fn test_corrcoef_with_zero_variance() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 1.0, 1.0, 1.0, 1.0];
        assert!(corrcoef(&x, &y)[0][1].is_nan());
        assert!(corrcoef(&x, &y)[1][0].is_nan());
    }

    #[test]
    fn test_skew() {
        let nums = vec![6, 6, 6, 9];
        assert_abs_diff_eq!(skew(&nums), 1.1547, epsilon = 0.001);

        let nums = [-1.0, -0.5, 1.0, 2.5];
        assert_abs_diff_eq!(skew(&nums), 0.3651, epsilon = 0.001);

        let nums = [1, 15];
        assert_eq!(skew(&nums), 0.0);
    }
}
