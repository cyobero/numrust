pub mod random;

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
/// let sequence = arange(0, 5, 1);
/// assert_eq!(sequence, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
///
/// let sequence = arange(0, 5, 2);
/// assert_eq!(sequence, vec![0.0, 2.0, 4.0]);
///
/// let sequence = arange(5, 0, -1);
/// assert_eq!(sequence, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
/// ```
///
/// # Panics
///
/// Panics if `step` is equal to zero.
pub fn arange(start: usize, stop: usize, step: isize) -> Vec<f64> {
    if step == 0 {
        panic!("Step size cannot be zero")
    };
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
    result
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
pub fn mean(nums: &[f64]) -> f64 {
    let sum: f64 = nums.iter().sum();
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
pub fn std_dev(nums: &[f64]) -> f64 {
    let var = variance(nums);
    if nums.len() == 0 {
        f64::NAN
    } else {
        var.sqrt()
    }
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

#[cfg(test)]
mod numrust_tests {
    use std::assert_eq;

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
        assert_eq!(arange(start, stop, step), expected_result);

        let start = 5;
        let stop = 20;
        let step = 5;

        let expected_result = vec![5.0, 10.0, 15.0];
        assert_eq!(arange(start, stop, step), expected_result);
    }

    #[test]
    #[should_panic(expected = "Step size cannot be zero")]
    fn test_arange_with_zero_step() {
        let start = 0;
        let stop = 10;
        let step = 0;

        let expected_result = vec![];
        assert_eq!(arange(start, stop, step), expected_result);
    }

    #[test]
    fn test_arange_with_negative_step() {
        let start = 10;
        let stop = 0;
        let step = -2;

        let expected_result = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert_eq!(arange(start, stop, step), expected_result);

        let seq = arange(5, 0, -1);
        assert_eq!(seq, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }
}
