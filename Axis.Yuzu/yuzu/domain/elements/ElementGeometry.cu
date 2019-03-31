#include "ElementGeometry.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

namespace ayde = axis::yuzu::domain::elements;
namespace aydi = axis::yuzu::domain::integration;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY ayde::ElementGeometry::ElementGeometry( void )
{
  // nothing to do here; private implementation
}

GPU_ONLY ayde::ElementGeometry::~ElementGeometry( void )
{
	// nothing to do here
}

GPU_ONLY void ayde::ElementGeometry::SetNode( int nodeIndex, const ayfm::RelativePointer& node )
{
  ayfm::RelativePointer * n = yabsptr<ayfm::RelativePointer>(nodes_);
  n[nodeIndex] = node;
}

GPU_ONLY const ayde::Node& ayde::ElementGeometry::GetNode( int nodeId ) const
{
  const ayfm::RelativePointer * n = yabsptr<ayfm::RelativePointer>(nodes_);
  return yabsref<Node>(n[nodeId]);
}

GPU_ONLY ayde::Node& ayde::ElementGeometry::GetNode( int nodeId )
{
  ayfm::RelativePointer * n = yabsptr<ayfm::RelativePointer>(nodes_);
  return yabsref<Node>(n[nodeId]);
}

GPU_ONLY const ayde::Node& ayde::ElementGeometry::operator[]( int nodeId ) const
{
  const ayfm::RelativePointer * n = yabsptr<ayfm::RelativePointer>(nodes_);
  return yabsref<Node>(n[nodeId]);
}

GPU_ONLY ayde::Node& ayde::ElementGeometry::operator[]( int nodeId )
{
  ayfm::RelativePointer * n = yabsptr<ayfm::RelativePointer>(nodes_);
  return yabsref<Node>(n[nodeId]);
}

GPU_ONLY int ayde::ElementGeometry::GetNodeCount( void ) const
{
	return numNodes_;
}

GPU_ONLY void ayde::ElementGeometry::ExtractLocalField( ayfb::ColumnVector& localField, 
                                               const ayfb::ColumnVector& globalField ) const
{
	// build a matrix large enough for local field	
	int totalDof = GetTotalDofCount();
	int pos = 0;
  const ayfm::RelativePointer * n = yabsptr<ayfm::RelativePointer>(nodes_);
	for (int i = 0; i < numNodes_; i++)
	{
    const Node& node = yabsref<Node>(n[i]);
		int nodeDofCount = node.GetDofCount();
		for (int j = 0; j < nodeDofCount; j++)
		{
			localField(pos) = globalField(node[j].GetId());
			++pos;
		}
	}
}

GPU_ONLY bool ayde::ElementGeometry::HasIntegrationPoints( void ) const
{
	return (numIntegrPoints_ > 0);
}

GPU_ONLY const aydi::IntegrationPoint& ayde::ElementGeometry::GetIntegrationPoint( int index ) const
{
  const ayfm::RelativePointer *points = yabsptr<ayfm::RelativePointer>(points_);
  return yabsref<aydi::IntegrationPoint>(points[index]);
}

GPU_ONLY aydi::IntegrationPoint& ayde::ElementGeometry::GetIntegrationPoint( int index )
{
  ayfm::RelativePointer *points = yabsptr<ayfm::RelativePointer>(points_);
  return yabsref<aydi::IntegrationPoint>(points[index]);
}

GPU_ONLY void ayde::ElementGeometry::SetIntegrationPoint( int index, const ayfm::RelativePointer& point )
{
  ayfm::RelativePointer *points = yabsptr<ayfm::RelativePointer>(points_);
  points[index] = point;
}

GPU_ONLY int ayde::ElementGeometry::GetIntegrationPointCount( void ) const
{
  return numIntegrPoints_;
}

GPU_ONLY bool ayde::ElementGeometry::HasNode( const ayde::Node& node ) const
{
  const ayfm::RelativePointer * n = yabsptr<ayfm::RelativePointer>(nodes_);
	for (int i = 0; i < numNodes_; ++i)
	{
		if (yabsptr<Node>(n[i]) == &node) return true;
	}
	return false;
}

GPU_ONLY int ayde::ElementGeometry::GetNodeIndex( const ayde::Node& node ) const
{
  const ayfm::RelativePointer * n = yabsptr<ayfm::RelativePointer>(nodes_);
	for (int i = 0; i < numNodes_; ++i)
	{
		if (yabsptr<Node>(n[i]) == &node) return i;
	}
	return -1;
}

GPU_ONLY int ayde::ElementGeometry::GetTotalDofCount( void ) const
{
  const ayfm::RelativePointer * n = yabsptr<ayfm::RelativePointer>(nodes_);
	int totalDof = 0;
	for (int i = 0; i < numNodes_; ++i)
	{
		totalDof += yabsref<Node>(n[i]).GetDofCount();
	}
	return totalDof;
}
