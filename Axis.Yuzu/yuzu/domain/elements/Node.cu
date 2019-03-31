#include "Node.hpp"
#include "yuzu/foundation/memory/pointer.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"
#include "ReverseConnectivityList.hpp"

namespace ayde = axis::yuzu::domain::elements;
namespace ayfm = axis::yuzu::foundation::memory;
namespace ayfb = axis::yuzu::foundation::blas;

GPU_ONLY ayde::Node::~Node(void)
{
  // nothing to do here
}

GPU_ONLY const coordtype& ayde::Node::X( void ) const
{
  return _x;
}

GPU_ONLY coordtype& ayde::Node::X( void )
{
  return _x;
}
GPU_ONLY const coordtype& ayde::Node::Y( void ) const
{
  return _y;
}

GPU_ONLY coordtype& ayde::Node::Y( void )
{
  return _y;
}
GPU_ONLY const coordtype& ayde::Node::Z( void ) const
{
  return _z;
}

GPU_ONLY coordtype& ayde::Node::Z( void )
{
  return _z;
}

GPU_ONLY coordtype& ayde::Node::CurrentX( void )
{
  return curX_;
}

GPU_ONLY coordtype ayde::Node::CurrentX( void ) const
{
  return curX_;
}

GPU_ONLY coordtype& ayde::Node::CurrentY( void )
{
  return curY_;
}

GPU_ONLY coordtype ayde::Node::CurrentY( void ) const
{
  return curY_;
}

GPU_ONLY coordtype& ayde::Node::CurrentZ( void )
{
  return curZ_;
}

GPU_ONLY coordtype ayde::Node::CurrentZ( void ) const
{
  return curZ_;
}
GPU_ONLY const ayde::DoF& ayde::Node::GetDoF( int index ) const
{
  return yabsref<DoF>(_dofs[index]);
}

GPU_ONLY ayde::DoF& ayde::Node::GetDoF( int index )
{
  return yabsref<DoF>(_dofs[index]);
}

GPU_ONLY int ayde::Node::GetDofCount( void ) const
{
  return _numDofs;
}

GPU_ONLY bool ayde::Node::WasInitialized( void ) const
{
  return GetDofCount() > 0;
}

GPU_ONLY const ayde::DoF& ayde::Node::operator[]( int index ) const
{
	return yabsref<DoF>(_dofs[index]);
}

GPU_ONLY ayde::DoF& ayde::Node::operator[]( int index )
{
  return yabsref<DoF>(_dofs[index]);
}

GPU_ONLY ayde::Node::id_type ayde::Node::GetUserId( void ) const
{
	return _externalId;
}

GPU_ONLY ayde::Node::id_type ayde::Node::GetInternalId( void ) const
{
	return _internalId;
}

GPU_ONLY int ayde::Node::GetConnectedElementCount( void ) const
{
  const ReverseConnectivityList& connList = yabsref<ReverseConnectivityList>(reverseConnList_);
	return (int)connList.Count();
}

GPU_ONLY ayde::FiniteElement& ayde::Node::GetConnectedElement( int elementIndex ) const
{
  const ReverseConnectivityList& connList = yabsref<ReverseConnectivityList>(reverseConnList_);
  ayfm::RelativePointer ptr = connList.GetItem(elementIndex);
  return yabsref<ayde::FiniteElement>(ptr);
}

GPU_ONLY ayfb::ColumnVector& ayde::Node::Strain( void )
{
  return yabsref<ayfb::ColumnVector>(_strain);
}

GPU_ONLY const ayfb::ColumnVector& ayde::Node::Strain( void ) const
{
  return yabsref<ayfb::ColumnVector>(_strain);
}

GPU_ONLY ayfb::ColumnVector& ayde::Node::Stress( void )
{
  return yabsref<ayfb::ColumnVector>(_stress);
}

GPU_ONLY const ayfb::ColumnVector& ayde::Node::Stress( void ) const
{
  return yabsref<ayfb::ColumnVector>(_stress);
}

GPU_ONLY void ayde::Node::ResetStrain( void )
{
	Strain().ClearAll();	
}

GPU_ONLY void ayde::Node::ResetStress( void )
{
	Stress().ClearAll();	
}
