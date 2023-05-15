pub mod random;

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
}
