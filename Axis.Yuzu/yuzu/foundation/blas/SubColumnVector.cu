#include "SubColumnVector.hpp"
#include <math.h>
#include "yuzu/foundation/memory/pointer.hpp"
#include "yuzu/foundation/blas/VectorView.hpp"

namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY ayfb::SubColumnVector::SubColumnVector( ayfm::RelativePointer& targetVector, 
                                                 const ayfm::RelativePointer& mask ) :
  targetVector_(targetVector), mask_(mask)
{
  // nothing to do here
}

GPU_ONLY ayfb::SubColumnVector::~SubColumnVector( void )
{
  // nothing to do here
}

GPU_ONLY real ayfb::SubColumnVector::GetElement( size_type pos ) const
{
  return yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(pos));
}

GPU_ONLY ayfb::SubColumnVector::self& ayfb::SubColumnVector::SetElement( size_type pos, real value )
{
  yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(pos)) = value;
  return *this;
}

GPU_ONLY ayfb::SubColumnVector::self& ayfb::SubColumnVector::Accumulate( size_type pos, real value )
{
  yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(pos)) += value;
  return *this;
}

GPU_ONLY ayfb::SubColumnVector::self& ayfb::SubColumnVector::CopyFrom( const self& vector )
{
  size_type len = vector.Length();
  for (size_type i = 0; i < len; i++)
  {
    yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i)) = vector(i);
  }
  return *this;
}

GPU_ONLY ayfb::SubColumnVector::self& ayfb::SubColumnVector::CopyFrom( const inner_type& vector )
{
  size_type len = Length();
  for (size_type i = 0; i < len; i++)
  {
    yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i)) = vector(i);
  }
  return *this;
}

GPU_ONLY ayfb::SubColumnVector::self& ayfb::SubColumnVector::ClearAll( void )
{
  SetAll(0);
  return *this;
}

GPU_ONLY ayfb::SubColumnVector::self& ayfb::SubColumnVector::Scale( real value )
{
  size_type len = Length();
  for (size_type i = 0; i < len; i++)
  {
    yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i)) *= value;
  }
  return *this;
}

GPU_ONLY ayfb::SubColumnVector::self& ayfb::SubColumnVector::SetAll( real value )
{
  size_type len = Length();
  for (size_type i = 0; i < len; i++)
  {
    yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i)) = value;
  }
  return *this;
}

GPU_ONLY size_type ayfb::SubColumnVector::Length( void ) const
{
  return yabsref<ayfb::VectorView>(mask_).Length();
}

GPU_ONLY real ayfb::SubColumnVector::SelfScalarProduct( void ) const
{
  size_type len = Length();
  real result = 0;
  for (size_type i = 0; i < len; i++)
  {
    real val = yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i));
    result += val*val;
  }
  return result;
}

GPU_ONLY real ayfb::SubColumnVector::Norm( void ) const
{
  return sqrt(SelfScalarProduct());
}

GPU_ONLY ayfb::SubColumnVector::self& ayfb::SubColumnVector::Invert( void )
{
  size_type len = Length();
  for (size_type i = 0; i < len; i++)
  {
    real val = yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i));
    yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(i)) = 1.0 / val;
  }
  return *this;
}

GPU_ONLY size_type ayfb::SubColumnVector::Rows( void ) const
{
  return 1;
}

GPU_ONLY size_type ayfb::SubColumnVector::Columns( void ) const
{
  return yabsref<ayfb::VectorView>(mask_).Length();
}

GPU_ONLY real ayfb::SubColumnVector::operator()( size_type pos ) const
{
  return yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(pos));
}

GPU_ONLY real& ayfb::SubColumnVector::operator()( size_type pos )
{
  return yabsref<inner_type>(targetVector_)(yabsref<ayfb::VectorView>(mask_)(pos));
}
