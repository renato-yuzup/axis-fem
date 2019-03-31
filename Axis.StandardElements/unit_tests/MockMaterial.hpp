#pragma once
#include "domain/materials/MaterialModel.hpp"
#if !defined(AXIS_NO_MEMORY_ARENA)
#include "foundation/memory/RelativePointer.hpp"
#endif

namespace axis { namespace unit_tests { namespace AxisStandardElements {

class MockMaterial : public axis::domain::materials::MaterialModel
{
public:
	MockMaterial(void);
	~MockMaterial(void);
	virtual void Destroy( void ) const;
  virtual MaterialModel& Clone( int numPoints ) const;
	virtual const axis::foundation::blas::DenseMatrix& 
    GetMaterialTensor( void ) const;
	virtual void UpdateStresses( 
    axis::domain::physics::UpdatedPhysicalState& updatedState,
    const axis::domain::physics::InfinitesimalState& currentState, 
    const axis::domain::analyses::AnalysisTimeline& timeInfo,
    int materialPointIndex);
	virtual real GetWavePropagationSpeed( void ) const;
  virtual real GetBulkModulus( void ) const;
  virtual real GetShearModulus( void ) const;

  virtual axis::foundation::uuids::Uuid GetTypeId( void ) const;

private:
#if defined(AXIS_NO_MEMORY_ARENA)
  axis::foundation::blas::Matrix *_materialMatrix;
#else
  axis::foundation::memory::RelativePointer _materialMatrix;
#endif
};

} } }
