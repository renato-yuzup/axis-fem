#if defined _DEBUG || defined DEBUG

#include "stdafx.h"
#include <exception>
#include "MockMaterial.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "domain/physics/UpdatedPhysicalState.hpp"
#include "foundation/blas/DenseMatrix.hpp"
#include "foundation/memory/pointer.hpp"

namespace ada = axis::domain::analyses;
namespace adm = axis::domain::materials;
namespace adp = axis::domain::physics;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;
namespace aus = axis::unit_tests::AxisStandardElements;

aus::MockMaterial::MockMaterial( void ) : MaterialModel(7850, 1)
{
	// create material matrix
	real E  = 200e9;
	real nu = 0.3;
	real rho = 7850;
  afm::RelativePointer ptr = afb::DenseMatrix::CreateFromGlobalMemory(6, 6);
  afb::DenseMatrix& D = absref<afb::DenseMatrix>(ptr);
	D.ClearAll();

	// matrix terms
	real c11 = E*(1-nu) / ((1-2*nu)*(1+nu));
	real c12 = E*nu / ((1-2*nu)*(1+nu));
	real G  = E / (2*(1+nu));

	D(0,0) = c11; D(0,1) = c12; D(0,2) = c12; 
	D(1,0) = c12; D(1,1) = c11; D(1,2) = c12; 
	D(2,0) = c12; D(2,1) = c12; D(2,2) = c11; 
	D(3,3) = G; D(4,4) = G; D(5,5) = G; 

  _materialMatrix = ptr;          
}

aus::MockMaterial::~MockMaterial( void )
{
  absref<afb::DenseMatrix>(_materialMatrix).Destroy();
  System::GlobalMemory().Deallocate(_materialMatrix);
}

void aus::MockMaterial::Destroy( void ) const
{
	delete this;
}

const afb::DenseMatrix& aus::MockMaterial::GetMaterialTensor( void ) const
{
  return absref<afb::DenseMatrix>(_materialMatrix);
}

adm::MaterialModel& aus::MockMaterial::Clone( int ) const
{
	return *new MockMaterial();
}

void aus::MockMaterial::UpdateStresses(adp::UpdatedPhysicalState& updatedState,
  const adp::InfinitesimalState& currentState, const ada::AnalysisTimeline&, int)
{
	afb::VectorProduct(updatedState.LastStressIncrement(), 1.0, 
    absref<afb::DenseMatrix>(_materialMatrix), 
    currentState.LastStrainIncrement());
	afb::VectorSum(updatedState.Stress(), currentState.Stress(), 1.0, 
    currentState.LastStressIncrement());
}

real aus::MockMaterial::GetWavePropagationSpeed( void ) const
{
	throw std::exception("The method or operation is not implemented.");
}


real aus::MockMaterial::GetBulkModulus( void ) const
{
  throw std::exception("The method or operation is not implemented.");
}

real aus::MockMaterial::GetShearModulus( void ) const
{
  throw std::exception("The method or operation is not implemented.");
}

axis::foundation::uuids::Uuid axis::unit_tests::AxisStandardElements::MockMaterial::GetTypeId( void ) const
{
  throw std::exception("The method or operation is not implemented.");
}

#endif
