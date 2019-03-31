#include "FiniteElement.hpp"
#include "yuzu/foundation/memory/pointer.hpp"
#include "ElementGeometry.hpp"

namespace ayde = axis::yuzu::domain::elements;
namespace aydp = axis::yuzu::domain::physics;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY ayde::FiniteElement::FiniteElement( void ) 
{
  // nothing to do here; private implementation
}

GPU_ONLY ayde::FiniteElement::~FiniteElement(void)
{
	// nothing to do here
}

GPU_ONLY id_type ayde::FiniteElement::GetInternalId( void ) const
{
  return internalId_;
}

GPU_ONLY id_type ayde::FiniteElement::GetUserId( void ) const
{
  return externalId_;
}

GPU_ONLY ayde::ElementGeometry& ayde::FiniteElement::Geometry( void )
{
  return yabsref<ayde::ElementGeometry>(geometry_);
}

GPU_ONLY const ayde::ElementGeometry& ayde::FiniteElement::Geometry( void ) const
{
  return yabsref<ayde::ElementGeometry>(geometry_);
}

GPU_ONLY aydp::InfinitesimalState& ayde::FiniteElement::PhysicalState( void )
{
  return yabsref<aydp::InfinitesimalState>(physicalState_);
}

GPU_ONLY void ayde::FiniteElement::ExtractLocalField(ayfb::ColumnVector& localField, 
                                                     const ayfb::ColumnVector& globalField) const
{
  const ayde::ElementGeometry& geometry = yabsref<ayde::ElementGeometry>(geometry_);
	return geometry.ExtractLocalField(localField, globalField);
}
