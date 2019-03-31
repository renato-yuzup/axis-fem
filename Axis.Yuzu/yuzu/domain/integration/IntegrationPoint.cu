#include "IntegrationPoint.hpp"
#include "yuzu/foundation/memory/pointer.hpp"

namespace aydi = axis::yuzu::domain::integration;
namespace aydp = axis::yuzu::domain::physics;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY aydi::IntegrationPoint::IntegrationPoint( void )
{
  // nothing to do here; private implementation
}

GPU_ONLY aydi::IntegrationPoint::~IntegrationPoint( void )
{
  // nothing to do here; private implementation
}

GPU_ONLY aydp::InfinitesimalState& aydi::IntegrationPoint::State( void )
{
  return yabsref<aydp::InfinitesimalState>(state_);
}

GPU_ONLY const aydp::InfinitesimalState& aydi::IntegrationPoint::State( void ) const
{
  return yabsref<aydp::InfinitesimalState>(state_);
}

GPU_ONLY real& aydi::IntegrationPoint::Weight( void )
{
  return weight_;
}

GPU_ONLY real aydi::IntegrationPoint::Weight( void ) const
{
  return weight_;
}

GPU_ONLY real& aydi::IntegrationPoint::X( void )
{
  return x_;
}

GPU_ONLY real aydi::IntegrationPoint::X( void ) const
{
  return x_;
}

GPU_ONLY real& aydi::IntegrationPoint::Y( void )
{
  return y_;
}

GPU_ONLY real aydi::IntegrationPoint::Y( void ) const
{
  return y_;
}

GPU_ONLY real& aydi::IntegrationPoint::Z( void )
{
  return z_;
}

GPU_ONLY real aydi::IntegrationPoint::Z( void ) const
{
  return z_;
}
