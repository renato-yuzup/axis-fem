#include "continuum.hpp"
#include "yuzu/foundation/blas/matrix_operations.hpp"

namespace aym  = axis::yuzu::mechanics;
namespace ayfb = axis::yuzu::foundation::blas;

GPU_ONLY void axis::yuzu::mechanics::CalculateFTimeDerivative( 
  axis::yuzu::foundation::blas::DenseMatrix& Fdiff, 
  const axis::yuzu::foundation::blas::DenseMatrix& Fnext, 
  const axis::yuzu::foundation::blas::DenseMatrix& Fcur, real dt )
{
  ayfb::Sum(Fdiff, 1.0, Fnext, -1.0, Fcur);
  Fdiff /= dt;
}
GPU_ONLY void axis::yuzu::mechanics::CalculateFTimeDerivative( 
  axis::yuzu::foundation::blas::AutoDenseMatrix<3,3>& Fdiff, 
  const axis::yuzu::foundation::blas::DenseMatrix& Fnext, 
  const axis::yuzu::foundation::blas::DenseMatrix& Fcur, real dt )
{
  ayfb::Sum(Fdiff, 1.0, Fnext, -1.0, Fcur);
  Fdiff /= dt;
}

GPU_ONLY void axis::yuzu::mechanics::CalculateVelocityGradient( 
  axis::yuzu::foundation::blas::DenseMatrix& Lcur, 
  const axis::yuzu::foundation::blas::DenseMatrix& Fnext, 
  const axis::yuzu::foundation::blas::DenseMatrix& Fcur, real dt )
{
  ayfb::DenseMatrix Fdiff(3, 3);
  CalculateFTimeDerivative(Fdiff, Fnext, Fcur, dt);
  CalculateVelocityGradient(Lcur, Fdiff, Fcur);
}
GPU_ONLY void axis::yuzu::mechanics::CalculateVelocityGradient( 
  axis::yuzu::foundation::blas::AutoDenseMatrix<3,3>& Lcur, 
  const axis::yuzu::foundation::blas::DenseMatrix& Fnext, 
  const axis::yuzu::foundation::blas::DenseMatrix& Fcur, real dt )
{
  ayfb::AutoDenseMatrix<3,3> Fdiff;
  CalculateFTimeDerivative(Fdiff, Fnext, Fcur, dt);
  CalculateVelocityGradient(Lcur, Fdiff, Fcur);
}

GPU_ONLY void axis::yuzu::mechanics::CalculateVelocityGradient( 
  axis::yuzu::foundation::blas::DenseMatrix& Lcur, 
  const axis::yuzu::foundation::blas::DenseMatrix& Fdiff, 
  const axis::yuzu::foundation::blas::DenseMatrix& Fcur )
{
  ayfb::DenseMatrix Finv(3, 3);
  ayfb::Inverse3D(Finv, Fcur);
  ayfb::Product(Lcur, 1.0, Fdiff, Finv);
}
GPU_ONLY void axis::yuzu::mechanics::CalculateVelocityGradient( 
  axis::yuzu::foundation::blas::AutoDenseMatrix<3,3>& Lcur, 
  const axis::yuzu::foundation::blas::AutoDenseMatrix<3,3>& Fdiff, 
  const axis::yuzu::foundation::blas::DenseMatrix& Fcur )
{
  ayfb::AutoDenseMatrix<3,3> Finv;
  ayfb::Inverse3D(Finv, Fcur);
  ayfb::Product(Lcur, 1.0, Fdiff, Finv);
}
