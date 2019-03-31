#include "Node.hpp"
#include "domain/elements/FiniteElement.hpp"

#include "foundation/OutOfBoundsException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/ArgumentException.hpp"
#include "System.hpp"
#include "foundation/memory/pointer.hpp"
#include "ReverseConnectivityList.hpp"

namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace afm = axis::foundation::memory;
namespace afb = axis::foundation::blas;

void ade::Node::initMembers(id_type internalId, id_type externalId, coordtype x, coordtype y, coordtype z)
{
	_internalId = internalId;
	_externalId = externalId;
	_x = x;
	_y = y;
	_z = z;
  curX_ = x;
  curY_ = y;
  curZ_ = z;
	_strain = NULLPTR;
	_stress = NULLPTR;
	_numDofs = 0;
  isConnectivityListLocked_ = false;
}

ade::Node::Node(id_type id) : _elements(new adc::ElementSet())
{
	initMembers(id, id, 0, 0, 0);
}

ade::Node::Node( id_type id, coordtype x, coordtype y, coordtype z ) : _elements(new adc::ElementSet())
{
	initMembers(id, id, x, y, z);
}

ade::Node::Node( id_type id, coordtype x, coordtype y ) : _elements(new adc::ElementSet())
{
	initMembers(id, id, x, y, 0);
}

ade::Node::Node( id_type internalId, id_type externalId ) : _elements(new adc::ElementSet())
{
	initMembers(internalId, externalId, 0, 0, 0);
}

ade::Node::Node( id_type internalId, id_type externalId, coordtype x, coordtype y ) : _elements(new adc::ElementSet())
{
	initMembers(internalId, externalId, x, y, 0);
}

ade::Node::Node( id_type internalId, id_type externalId, coordtype x, coordtype y, coordtype z ) : _elements(new adc::ElementSet())
{
	initMembers(internalId, externalId, x, y, z);
}

ade::Node::~Node(void)
{
	for (int i = 0; i < _numDofs; i++)
	{
		absref<DoF>(_dofs[i]).Destroy();
	}
	_elements->Destroy();

#if defined(AXIS_NO_MEMORY_ARENA)
	if (_strain != NULLPTR) delete _strain;
	if (_stress != NULLPTR) delete _stress;
#else
  if (_strain != NULLPTR)
  {
    Strain().Destroy();
    System::ModelMemory().Deallocate(_strain);
  }
  if (_stress != NULLPTR)
  {
    Stress().Destroy();
    System::ModelMemory().Deallocate(_stress);
  }
#endif
}

const ade::DoF& ade::Node::operator[]( int index ) const
{
	return absref<DoF>(_dofs[index]);
}

ade::DoF& ade::Node::operator[]( int index )
{
  return absref<DoF>(_dofs[index]);
}

ade::Node::id_type ade::Node::GetUserId( void ) const
{
	return _externalId;
}

ade::Node::id_type ade::Node::GetInternalId( void ) const
{
	return _internalId;
}

void ade::Node::ConnectElement( afm::RelativePointer& element )
{  
  if (isConnectivityListLocked_)
  {
    throw axis::foundation::InvalidOperationException();
  }
	_elements->Add(element);
}

int ade::Node::GetConnectedElementCount( void ) const
{
	return (int)_elements->Count();
}

void ade::Node::CompileConnectivityList( void )
{
  if (isConnectivityListLocked_) return;
  isConnectivityListLocked_ = true;
  int count = _elements->Count();
  reverseConnList_ = ReverseConnectivityList::Create(count);
  ReverseConnectivityList& connList = absref<ReverseConnectivityList>(reverseConnList_);
  for (int i = 0; i < count; ++i)
  {
    connList.SetItem(i, _elements->GetPointerByPosition(i));
  }
}

ade::FiniteElement& ade::Node::GetConnectedElement( int elementIndex ) const
{
  if (isConnectivityListLocked_)
  {
    const ReverseConnectivityList& connList = absref<ReverseConnectivityList>(reverseConnList_);
    afm::RelativePointer ptr = connList.GetItem(elementIndex);
    return absref<ade::FiniteElement>(ptr);
  }
  if (elementIndex >= _elements->Count() || elementIndex < 0)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return _elements->GetByPosition(elementIndex);
}

const coordtype& ade::Node::X( void ) const
{
	return _x;
}

coordtype& ade::Node::X( void )
{
	return _x;
}
const coordtype& ade::Node::Y( void ) const
{
	return _y;
}

coordtype& ade::Node::Y( void )
{
	return _y;
}
const coordtype& ade::Node::Z( void ) const
{
	return _z;
}

coordtype& ade::Node::Z( void )
{
	return _z;
}

void ade::Node::Destroy( void ) const
{
	delete this;
}

void ade::Node::InitDofs( int dofCount, DoF::id_type startId )
{
	if (_numDofs != 0)
	{
		throw axis::foundation::InvalidOperationException(_T("Cannot re-initialize dofs."));
	}
	if (dofCount <= 0 || dofCount > 6)
	{
		throw axis::foundation::ArgumentException(_T("Invalid dof count"));
	}
	if (startId < 0)
	{
		throw axis::foundation::ArgumentException(_T("Invalid dof id"));
	}

	DoF::id_type dofId = startId;
	_numDofs = dofCount;
  afm::RelativePointer myPtr = afm::RelativePointer::FromAbsolute(this, afm::RelativePointer::kModelMemory);
	for (int i = 0; i < dofCount; ++i)
	{
		_dofs[i] = DoF::Create(dofId, i, myPtr);
		++dofId;
	}
}

afb::ColumnVector& ade::Node::Strain( void )
{
	if (_strain == NULLPTR)
	{
		throw axis::foundation::InvalidOperationException(_T("Nodal strain not initialized."));
	}
#if defined(AXIS_NO_MEMORY_ARENA)
	return *_strain;
#else
  return *(afb::ColumnVector *)*_strain;
#endif
}

const afb::ColumnVector& ade::Node::Strain( void ) const
{
	if (_strain == NULLPTR)
	{
		throw axis::foundation::InvalidOperationException(_T("Nodal strain not initialized."));
	}
#if defined(AXIS_NO_MEMORY_ARENA)
  return *_strain;
#else
  return *(afb::ColumnVector *)*_strain;
#endif
}

afb::ColumnVector& ade::Node::Stress( void )
{
	if (_stress == NULLPTR)
	{
		throw axis::foundation::InvalidOperationException(_T("Nodal stress not initialized."));
	}
#if defined(AXIS_NO_MEMORY_ARENA)
  return *_stress;
#else
  return *(afb::ColumnVector *)*_stress;
#endif
}

const afb::ColumnVector& ade::Node::Stress( void ) const
{
	if (_stress == NULLPTR)
	{
		throw axis::foundation::InvalidOperationException(_T("Nodal stress not initialized."));
	}
#if defined(AXIS_NO_MEMORY_ARENA)
  return *_stress;
#else
  return *(afb::ColumnVector *)*_stress;
#endif
}

void ade::Node::ResetStrain( void )
{
	if (_strain == NULLPTR)
  {
#if defined(AXIS_NO_MEMORY_ARENA)
    _strain = new afb::ColumnVector(6);
#else
    _strain = afb::ColumnVector::Create(6);
#endif
  }
	Strain().ClearAll();	
}

void ade::Node::ResetStress( void )
{
	if (_stress == NULLPTR)
  {
#if defined(AXIS_NO_MEMORY_ARENA)
    _stress = new afb::ColumnVector(6);
#else
    _stress = afb::ColumnVector::Create(6);
#endif
  } 
	Stress().ClearAll();	
}

bool ade::Node::WasInitialized( void ) const
{
	return GetDofCount() > 0;
}

const ade::DoF& ade::Node::GetDoF( int index ) const
{
	int n = _numDofs;
	if (index < 0 || index >= n)
	{
		throw axis::foundation::OutOfBoundsException();
	}

	return absref<DoF>(_dofs[index]);
}

ade::DoF& ade::Node::GetDoF( int index )
{
  int n = _numDofs;
  if (index < 0 || index >= n)
  {
    throw axis::foundation::OutOfBoundsException();
  }

  return absref<DoF>(_dofs[index]);
}

int ade::Node::GetDofCount( void ) const
{
	return _numDofs;
}

afm::RelativePointer ade::Node::Create( id_type id )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(Node));
  new (*ptr) Node(id);
  return ptr;
}
afm::RelativePointer ade::Node::Create( id_type internalId, id_type externalId )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(Node));
  new (*ptr) Node(internalId, externalId);
  return ptr;
}
afm::RelativePointer ade::Node::Create( id_type id, coordtype x, coordtype y, 
                                        coordtype z )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(Node));
  new (*ptr) Node(id, x, y, z);
  return ptr;
}
afm::RelativePointer ade::Node::Create( id_type id, coordtype x, coordtype y )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(Node));
  new (*ptr) Node(id, x, y);
  return ptr;
}
afm::RelativePointer ade::Node::Create( id_type internalId, id_type externalId, 
                                        coordtype x, coordtype y )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(Node));
  new (*ptr) Node(internalId, externalId, x, y);
  return ptr;
}
afm::RelativePointer ade::Node::Create( id_type internalId, id_type externalId, 
                                        coordtype x, coordtype y, coordtype z )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(Node));
  new (*ptr) Node(internalId, externalId, x, y, z);
  return ptr;
}

void * ade::Node::operator new( size_t bytes )
{
  // it is assumed that the Node object will remain in memory until the
  // program dies. That's why we discard the relative pointer
  afm::RelativePointer ptr = System::GlobalMemory().Allocate(bytes);
  return *ptr;
}

void * ade::Node::operator new( size_t, void *ptr )
{
  return ptr;
}

void ade::Node::operator delete( void * )
{
  // Since we have discarded the relative pointer, there is no way we
  // can delete the node. Nothing to do here.
}

void ade::Node::operator delete( void *, void * )
{
  // Since we have discarded the relative pointer, there is no way we
  // can delete the node. Nothing to do here.
}

coordtype& ade::Node::CurrentX( void )
{
  return curX_;
}

coordtype ade::Node::CurrentX( void ) const
{
  return curX_;
}

coordtype& ade::Node::CurrentY( void )
{
  return curY_;
}

coordtype ade::Node::CurrentY( void ) const
{
  return curY_;
}

coordtype& ade::Node::CurrentZ( void )
{
  return curZ_;
}

coordtype ade::Node::CurrentZ( void ) const
{
  return curZ_;
}
