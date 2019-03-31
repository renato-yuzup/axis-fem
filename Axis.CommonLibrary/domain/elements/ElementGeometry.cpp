#include "ElementGeometry.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "System.hpp"
#include "foundation/memory/pointer.hpp"

namespace ade = axis::domain::elements;
namespace adi = axis::domain::integration;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

ade::ElementGeometry::ElementGeometry( int numNodes )
{
	InitGeometry(numNodes);
  points_ = NULLPTR;
}

ade::ElementGeometry::ElementGeometry( int numNodes, int numIntegrationPoints )
{
	InitGeometry(numNodes);
  numIntegrPoints_ = numIntegrationPoints;
	points_ = System::ModelMemory().Allocate(sizeof(afm::RelativePointer)*numIntegrationPoints);
}

void ade::ElementGeometry::InitGeometry( int numNodes )
{
	// check for a valid number of nodes
	if (numNodes < 1)
	{
		throw axis::foundation::ArgumentException(_TEXT("numNodes"));
	}

	// create arrays
#if defined(AXIS_NO_MEMORY_ARENA)
  nodes_ = new Node*[numNodes];
  Node **nodes = nodes_;
#else
	nodes_ = System::ModelMemory().Allocate(sizeof(afm::RelativePointer) * numNodes);
  afm::RelativePointer *nodes = (afm::RelativePointer*)*nodes_;
#endif
	numNodes_ = numNodes;
	for (int i = 0; i < numNodes; i++)
	{
		nodes[i] = NULLPTR;
	}

	// init variables
	numIntegrPoints_ = 0;
}

ade::ElementGeometry::~ElementGeometry( void )
{
	if(numIntegrPoints_ > 0)
	{
    afm::RelativePointer *ptr = (afm::RelativePointer *)*points_;
    for (int i = 0; i < numIntegrPoints_; i++)
    {
      adi::IntegrationPoint& p = absref<adi::IntegrationPoint>(ptr[i]);
      System::ModelMemory().Deallocate(ptr[i]);
      p.Destroy();
    }
    System::ModelMemory().Deallocate(points_);
	}
#if defined(AXIS_NO_MEMORY_ARENA)
	delete [] nodes_;
#else
  System::ModelMemory().Deallocate(nodes_);
#endif
}

#if defined(AXIS_NO_MEMORY_ARENA)
void ade::ElementGeometry::SetNode( int nodeIndex, Node& node )
#else
void ade::ElementGeometry::SetNode( int nodeIndex, const afm::RelativePointer& node )
#endif
{
	if (nodeIndex < 0 || nodeIndex >= numNodes_)
	{
		throw axis::foundation::OutOfBoundsException();
	}
#if defined(AXIS_NO_MEMORY_ARENA)
  nodes_[nodeIndex] = &node;
#else
  afm::RelativePointer * n = (afm::RelativePointer*)*nodes_;
  n[nodeIndex] = node;
#endif
}

const ade::Node& ade::ElementGeometry::GetNode( int nodeId ) const
{
#if defined(AXIS_NO_MEMORY_ARENA)
  return *nodes_[nodeId];
#else
  const afm::RelativePointer * n = absptr<afm::RelativePointer>(nodes_);
  return absref<Node>(n[nodeId]);
#endif
}

ade::Node& ade::ElementGeometry::GetNode( int nodeId )
{
#if defined(AXIS_NO_MEMORY_ARENA)
  return *nodes_[nodeId];
#else
  afm::RelativePointer * n = absptr<afm::RelativePointer>(nodes_);
  return absref<Node>(n[nodeId]);
#endif
}

const ade::Node& ade::ElementGeometry::operator[]( int nodeId ) const
{
#if defined(AXIS_NO_MEMORY_ARENA)
  return *nodes_[nodeId];
#else
  const afm::RelativePointer * n = absptr<afm::RelativePointer>(nodes_);
  return absref<Node>(n[nodeId]);
#endif
}

ade::Node& ade::ElementGeometry::operator[]( int nodeId )
{
#if defined(AXIS_NO_MEMORY_ARENA)
  return *nodes_[nodeId];
#else
  afm::RelativePointer * n = absptr<afm::RelativePointer>(nodes_);
  return absref<Node>(n[nodeId]);
#endif
}

int ade::ElementGeometry::GetNodeCount( void ) const
{
	return numNodes_;
}

void ade::ElementGeometry::ExtractLocalField( afb::ColumnVector& localField, 
                                              const afb::ColumnVector& globalField ) const
{
	// build a matrix large enough for local field	
	int totalDof = GetTotalDofCount();
	int pos = 0;
#if defined(AXIS_NO_MEMORY_ARENA)
  Node** n = nodes_;
#else
  const afm::RelativePointer * n = absptr<afm::RelativePointer>(nodes_);
#endif

	for (int i = 0; i < numNodes_; i++)
	{
    const Node& node = absref<Node>(n[i]);
		int nodeDofCount = node.GetDofCount();
		for (int j = 0; j < nodeDofCount; j++)
		{
			localField(pos) = globalField(node[j].GetId());
			++pos;
		}
	}
}

bool ade::ElementGeometry::HasIntegrationPoints( void ) const
{
	return (numIntegrPoints_ > 0);
}

const adi::IntegrationPoint& ade::ElementGeometry::GetIntegrationPoint( int index ) const
{
  if (index < 0 || index >= numIntegrPoints_)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  afm::RelativePointer *points = (afm::RelativePointer *)*points_;
  return absref<adi::IntegrationPoint>(points[index]);
}

adi::IntegrationPoint& ade::ElementGeometry::GetIntegrationPoint( int index )
{
  if (index < 0 || index >= numIntegrPoints_)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  afm::RelativePointer *points = (afm::RelativePointer *)*points_;
  return absref<adi::IntegrationPoint>(points[index]);
}

void ade::ElementGeometry::SetIntegrationPoint( int index, const afm::RelativePointer& point )
{
  if (index < 0 || index >= numIntegrPoints_)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  afm::RelativePointer *points = (afm::RelativePointer *)*points_;
  points[index] = point;
}

int ade::ElementGeometry::GetIntegrationPointCount( void ) const
{
  return numIntegrPoints_;
}

bool ade::ElementGeometry::HasNode( const ade::Node& node ) const
{
#if defined(AXIS_NO_MEMORY_ARENA)
  Node** n = nodes_;
#else
  afm::RelativePointer * n = (afm::RelativePointer*)*nodes_;
#endif
	for (int i = 0; i < numNodes_; ++i)
	{
		if (absptr<Node>(n[i]) == &node) return true;
	}
	return false;
}

int ade::ElementGeometry::GetNodeIndex( const ade::Node& node ) const
{
#if defined(AXIS_NO_MEMORY_ARENA)
  Node** n = nodes_;
#else
  afm::RelativePointer * n = (afm::RelativePointer*)*nodes_;
#endif
	for (int i = 0; i < numNodes_; ++i)
	{
		if (absptr<Node>(n[i]) == &node) return i;
	}
	return -1;
}

int ade::ElementGeometry::GetTotalDofCount( void ) const
{
#if defined(AXIS_NO_MEMORY_ARENA)
  Node** n = nodes_;
#else
  afm::RelativePointer * n = (afm::RelativePointer*)*nodes_;
#endif
	int totalDof = 0;
	for (int i = 0; i < numNodes_; ++i)
	{
		totalDof += absref<Node>(n[i]).GetDofCount();
	}
	return totalDof;
}

#if !defined(AXIS_NO_MEMORY_ARENA)

afm::RelativePointer ade::ElementGeometry::Create( int numNodes, int integrationPointCount )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(ElementGeometry));
  new (*ptr) ElementGeometry(numNodes, integrationPointCount);
  return ptr;
}

afm::RelativePointer ade::ElementGeometry::Create( int numNodes )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(ElementGeometry));
  new (*ptr) ElementGeometry(numNodes);
  return ptr;
}

void * ade::ElementGeometry::operator new( size_t bytes )
{
  // It is supposed that the finite element object will remain in memory
  // until the end of the program. That's why we discard the relative
  // pointer. We ignore the fact that an exception might occur in
  // constructor because if it does happen, the program will end.
  afm::RelativePointer ptr = System::GlobalMemory().Allocate(bytes);
  return *ptr;
}

void * ade::ElementGeometry::operator new( size_t, void *ptr )
{
  return ptr;
}

void ade::ElementGeometry::operator delete( void * )
{
  // Since the relative pointer was discarded, we can't discard memory.
  // If it is really necessary, to free up resources, obliterating
  // memory pool is a better approach.
}

void ade::ElementGeometry::operator delete( void *, void * )
{
  // Since the relative pointer was discarded, we can't discard memory.
  // If it is really necessary, to free up resources, obliterating
  // memory pool is a better approach.
}

#endif
