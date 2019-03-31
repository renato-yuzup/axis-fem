#include "continuum.hpp"
#include "foundation/blas/matrix_operations.hpp"

namespace am = axis::mechanics;
namespace afb = axis::foundation::blas;

void am::CalculateFTimeDerivative( afb::DenseMatrix& Fdiff, 
  const afb::DenseMatrix& Fnext, const afb::DenseMatrix& Fcur, real dt )
{
  afb::Sum(Fdiff, 1.0, Fnext, -1.0, Fcur);
  Fdiff.Scale(1.0 / dt);
}

void am::CalculateVelocityGradient( afb::DenseMatrix& Lcur, 
  const afb::DenseMatrix& Fnext, const afb::DenseMatrix& Fcur, real dt )
{
  afb::DenseMatrix Fdiff(3, 3);
  CalculateFTimeDerivative(Fdiff, Fnext, Fcur, dt);
  CalculateVelocityGradient(Lcur, Fdiff, Fcur);
}

void am::CalculateVelocityGradient( afb::DenseMatrix& Lcur, 
  const afb::DenseMatrix& Fdiff, const afb::DenseMatrix& Fcur )
{
  afb::DenseMatrix Finv(3, 3);
  afb::Inverse3D(Finv, Fcur);
  afb::Product(Lcur, 1.0, Fdiff, Finv);
}
