#pragma once
#include "foundation/axis.Saintpaulia.hpp"
#include "foundation/blas/DenseMatrix.hpp"

namespace axis { namespace mechanics {

/**
 * Numerically calculates the time derivative of the deformation gradient,
 * for sufficient small time increments where errors due to approximation is
 * negligible.
 *
 * @param [in,out] Fdiff Output for the derivative of current deformation
 *                       gradient.
 * @param Fnext          Deformation gradient of the next time step.
 * @param Fcur           Current deformation gradient.
 * @param dt             Time increment.
 */
AXISSAINTPAULIA_API void CalculateFTimeDerivative(
  axis::foundation::blas::DenseMatrix& Fdiff,
  const axis::foundation::blas::DenseMatrix& Fnext,
  const axis::foundation::blas::DenseMatrix& Fcur,
  real dt);

/**
 * Numerically calculates the velocity gradient from the deformation gradient
 * of two consecutive states.
 *
 * @param [in,out] Lcur Output for the velocity gradient of current time step.
 * @param Fnext         Deformation gradient of the next time step.
 * @param Fcur          Deformation gradient of current time step.
 * @param dt            Time increment.
 */
AXISSAINTPAULIA_API void CalculateVelocityGradient(
  axis::foundation::blas::DenseMatrix& Lcur,
  const axis::foundation::blas::DenseMatrix& Fnext,
  const axis::foundation::blas::DenseMatrix& Fcur,
  real dt);

/**
 * Calculates the velocity gradient from the deformation gradient and its
 * corresponding time derivative.
 *
 * @param [in,out] Lcur Output for the velocity gradient.
 * @param Fdiff         Time derivative of deformation gradient.
 * @param Fcur          Deformation gradient.
 */
AXISSAINTPAULIA_API void CalculateVelocityGradient(
  axis::foundation::blas::DenseMatrix& Lcur,
  const axis::foundation::blas::DenseMatrix& Fdiff,
  const axis::foundation::blas::DenseMatrix& Fcur);


} } // namespace axis::mechanics
