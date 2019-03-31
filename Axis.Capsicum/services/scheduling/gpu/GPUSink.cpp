#include "GPUSink.hpp"

namespace assg = axis::services::scheduling::gpu;
namespace asmm = axis::services::messaging;

assg::GPUSink::GPUSink(void)
{
  softFailEnabled_ = false;
}

assg::GPUSink::~GPUSink(void)
{
  // nothing to do here
}

void assg::GPUSink::Notify( asmm::EventMessage& message )
{
  DispatchMessage(message);
  bool shouldFail = message.IsError() && !softFailEnabled_;
  if (shouldFail)
  {
    Fail();
  }
}

void assg::GPUSink::ToggleSoftFail( bool state )
{
  softFailEnabled_ = state;
}
