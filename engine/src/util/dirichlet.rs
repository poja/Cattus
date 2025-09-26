// Copyright 2018 Developers of the Rand project.
// Copyright 2013 The Rust Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The dirichlet distribution `Dirichlet(α₁, α₂, ..., αₙ)`.

use core::fmt;
use num_traits::{Float, NumCast};
use rand::Rng;
use rand_distr::{Beta, Distribution, Exp1, Gamma, Open01, StandardNormal};

/// A standard abstraction for distributions with multi-dimensional results
///
/// Implementations may also implement `Distribution<Vec<T>>`.
pub trait MultiDistribution<T> {
    /// The length of a sample (dimension of the distribution)
    fn sample_len(&self) -> usize;

    /// Sample a multi-dimensional result from the distribution
    ///
    /// The result is written to `output`. Implementations should assert that
    /// `output.len()` equals the result of [`Self::sample_len`].
    fn sample_to_slice<R: Rng + ?Sized>(&self, rng: &mut R, output: &mut [T]);
}

#[derive(Clone, Debug, PartialEq)]
struct DirichletFromGamma<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    samplers: Vec<Gamma<F>>,
}

/// Error type returned from [`DirchletFromGamma::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirichletFromGammaError {
    /// Gamma::new(a, 1) failed.
    GammmaNewFailed,
}

impl<F> DirichletFromGamma<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    /// Construct a new `DirichletFromGamma` with the given parameters `alpha`.
    ///
    /// This function is part of a private implementation detail.
    /// It assumes that the input is correct, so no validation of alpha is done.
    #[inline]
    fn new(alpha: &[F]) -> Result<DirichletFromGamma<F>, DirichletFromGammaError> {
        let mut gamma_dists = Vec::new();
        for a in alpha {
            let dist = Gamma::new(*a, F::one()).map_err(|_| DirichletFromGammaError::GammmaNewFailed)?;
            gamma_dists.push(dist);
        }
        Ok(DirichletFromGamma { samplers: gamma_dists })
    }
}

impl<F> MultiDistribution<F> for DirichletFromGamma<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    #[inline]
    fn sample_len(&self) -> usize {
        self.samplers.len()
    }
    fn sample_to_slice<R: Rng + ?Sized>(&self, rng: &mut R, output: &mut [F]) {
        assert_eq!(output.len(), self.sample_len());

        let mut sum = F::zero();

        for (s, g) in output.iter_mut().zip(self.samplers.iter()) {
            *s = g.sample(rng);
            sum = sum + *s;
        }
        let invacc = F::one() / sum;
        for s in output.iter_mut() {
            *s = *s * invacc;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct DirichletFromBeta<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    samplers: Box<[Beta<F>]>,
}

/// Error type returned from [`DirchletFromBeta::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirichletFromBetaError {
    /// Beta::new(a, b) failed.
    BetaNewFailed,
}

impl<F> DirichletFromBeta<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    /// Construct a new `DirichletFromBeta` with the given parameters `alpha`.
    ///
    /// This function is part of a private implementation detail.
    /// It assumes that the input is correct, so no validation of alpha is done.
    #[inline]
    fn new(alpha: &[F]) -> Result<DirichletFromBeta<F>, DirichletFromBetaError> {
        // `alpha_rev_csum` is the reverse of the cumulative sum of the
        // reverse of `alpha[1..]`.  E.g. if `alpha = [a0, a1, a2, a3]`, then
        // `alpha_rev_csum` is `[a1 + a2 + a3, a2 + a3, a3]`.
        // Note that instances of DirichletFromBeta will always have N >= 2,
        // so the subtractions of 1, 2 and 3 from N in the following are safe.
        let n = alpha.len();
        let mut alpha_rev_csum = vec![alpha[n - 1]; n - 1];
        for k in 0..(n - 2) {
            alpha_rev_csum[n - 3 - k] = alpha_rev_csum[n - 2 - k] + alpha[n - 2 - k];
        }

        // Zip `alpha[..(N-1)]` and `alpha_rev_csum`; for the example
        // `alpha = [a0, a1, a2, a3]`, the zip result holds the tuples
        // `[(a0, a1+a2+a3), (a1, a2+a3), (a2, a3)]`.
        // Then pass each tuple to `Beta::new()` to create the `Beta`
        // instances.
        let mut beta_dists = Vec::new();
        for (&a, &b) in alpha[..(n - 1)].iter().zip(alpha_rev_csum.iter()) {
            let dist = Beta::new(a, b).map_err(|_| DirichletFromBetaError::BetaNewFailed)?;
            beta_dists.push(dist);
        }
        Ok(DirichletFromBeta {
            samplers: beta_dists.into_boxed_slice(),
        })
    }
}

impl<F> MultiDistribution<F> for DirichletFromBeta<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    #[inline]
    fn sample_len(&self) -> usize {
        self.samplers.len() + 1
    }
    fn sample_to_slice<R: Rng + ?Sized>(&self, rng: &mut R, output: &mut [F]) {
        assert_eq!(output.len(), self.sample_len());

        let mut acc = F::one();

        for (s, beta) in output.iter_mut().zip(self.samplers.iter()) {
            let beta_sample = beta.sample(rng);
            *s = acc * beta_sample;
            acc = acc * (F::one() - beta_sample);
        }
        output[output.len() - 1] = acc;
    }
}

#[derive(Clone, Debug, PartialEq)]
enum DirichletRepr<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    /// Dirichlet distribution that generates samples using the Gamma distribution.
    FromGamma(DirichletFromGamma<F>),

    /// Dirichlet distribution that generates samples using the Beta distribution.
    FromBeta(DirichletFromBeta<F>),
}

/// The [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) `Dirichlet(α₁, α₂, ..., αₖ)`.
///
/// The Dirichlet distribution is a family of continuous multivariate
/// probability distributions parameterized by a vector of positive
/// real numbers `α₁, α₂, ..., αₖ`, where `k` is the number of dimensions
/// of the distribution. The distribution is supported on the `k-1`-dimensional
/// simplex, which is the set of points `x = [x₁, x₂, ..., xₖ]` such that
/// `0 ≤ xᵢ ≤ 1` and `∑ xᵢ = 1`.
/// It is a multivariate generalization of the [`Beta`](crate::Beta) distribution.
/// The distribution is symmetric when all `αᵢ` are equal.
///
/// # Plot
///
/// The following plot illustrates the 2-dimensional simplices for various
/// 3-dimensional Dirichlet distributions.
///
/// ![Dirichlet distribution](https://raw.githubusercontent.com/rust-random/charts/main/charts/dirichlet.png)
///
/// # Example
///
/// ```ignore
/// use rand::prelude::*;
/// use rand_distr::multi::Dirichlet;
/// use rand_distr::multi::MultiDistribution;
///
/// let dirichlet = Dirichlet::new(&[1.0, 2.0, 3.0]).unwrap();
/// let samples = dirichlet.sample(&mut rand::rng());
/// println!("{:?} is from a Dirichlet(&[1.0, 2.0, 3.0]) distribution", samples);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Dirichlet<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    repr: DirichletRepr<F>,
}

/// Error type returned from [`Dirichlet::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// `alpha.len() < 2`.
    AlphaTooShort,
    /// `alpha <= 0.0` or `nan`.
    AlphaTooSmall,
    /// `alpha` is subnormal.
    /// Variate generation methods are not reliable with subnormal inputs.
    AlphaSubnormal,
    /// `alpha` is infinite.
    AlphaInfinite,
    /// Failed to create required Gamma distribution(s).
    FailedToCreateGamma,
    /// Failed to create required Beta distribition(s).
    FailedToCreateBeta,
    /// `size < 2`.
    SizeTooSmall,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Error::AlphaTooShort | Error::SizeTooSmall => "less than 2 dimensions in Dirichlet distribution",
            Error::AlphaTooSmall => "alpha is not positive in Dirichlet distribution",
            Error::AlphaSubnormal => "alpha contains a subnormal value in Dirichlet distribution",
            Error::AlphaInfinite => "alpha contains an infinite value in Dirichlet distribution",
            Error::FailedToCreateGamma => "failed to create required Gamma distribution for Dirichlet distribution",
            Error::FailedToCreateBeta => "failed to create required Beta distribution for Dirichlet distribution",
        })
    }
}

impl std::error::Error for Error {}

impl<F> Dirichlet<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    /// Construct a new `Dirichlet` with the given alpha parameter `alpha`.
    ///
    /// Requires `alpha.len() >= 2`, and each value in `alpha` must be positive,
    /// finite and not subnormal.
    #[inline]
    pub fn new(alpha: &[F]) -> Result<Dirichlet<F>, Error> {
        if alpha.len() < 2 {
            return Err(Error::AlphaTooShort);
        }
        for &ai in alpha.iter() {
            if !(ai > F::zero()) {
                // This also catches nan.
                return Err(Error::AlphaTooSmall);
            }
            if ai.is_infinite() {
                return Err(Error::AlphaInfinite);
            }
            if !ai.is_normal() {
                return Err(Error::AlphaSubnormal);
            }
        }

        if alpha.iter().all(|&x| x <= NumCast::from(0.1).unwrap()) {
            // Use the Beta method when all the alphas are less than 0.1  This
            // threshold provides a reasonable compromise between using the faster
            // Gamma method for as wide a range as possible while ensuring that
            // the probability of generating nans is negligibly small.
            let dist = DirichletFromBeta::new(alpha).map_err(|_| Error::FailedToCreateBeta)?;
            Ok(Dirichlet {
                repr: DirichletRepr::FromBeta(dist),
            })
        } else {
            let dist = DirichletFromGamma::new(alpha).map_err(|_| Error::FailedToCreateGamma)?;
            Ok(Dirichlet {
                repr: DirichletRepr::FromGamma(dist),
            })
        }
    }
}

impl<F> MultiDistribution<F> for Dirichlet<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    #[inline]
    fn sample_len(&self) -> usize {
        match &self.repr {
            DirichletRepr::FromGamma(dirichlet) => dirichlet.sample_len(),
            DirichletRepr::FromBeta(dirichlet) => dirichlet.sample_len(),
        }
    }
    fn sample_to_slice<R: Rng + ?Sized>(&self, rng: &mut R, output: &mut [F]) {
        match &self.repr {
            DirichletRepr::FromGamma(dirichlet) => dirichlet.sample_to_slice(rng, output),
            DirichletRepr::FromBeta(dirichlet) => dirichlet.sample_to_slice(rng, output),
        }
    }
}

impl<F> Distribution<Vec<F>> for Dirichlet<F>
where
    F: Float + Default,
    StandardNormal: Distribution<F>,
    Exp1: Distribution<F>,
    Open01: Distribution<F>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<F> {
        let mut buf = vec![Default::default(); self.sample_len()];
        self.sample_to_slice(rng, &mut buf);
        buf
    }
}
