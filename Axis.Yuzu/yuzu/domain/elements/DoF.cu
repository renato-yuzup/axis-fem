#include "DoF.hpp"
#include "Node.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

namespace ayde = axis::yuzu::domain::elements;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY ayde::DoF::DoF(void)
{
  // nothing to do here
}

GPU_ONLY ayde::DoF::~DoF(void)
{
	// nothing to do here
}

GPU_ONLY id_type ayde::DoF::GetId( void ) const
{
	return _id;
}

GPU_ONLY bool ayde::DoF::HasBoundaryConditionApplied( void ) const
{
	return _condition != NULL;
}

GPU_ONLY ayde::Node& ayde::DoF::GetParentNode( void )
{
	return yabsref<Node>(_parentNode);
}

GPU_ONLY const ayde::Node& ayde::DoF::GetParentNode( void ) const
{
	return yabsref<Node>(_parentNode);
}

GPU_ONLY int ayde::DoF::GetLocalIndex( void ) const
{
	return _localIndex;
}
