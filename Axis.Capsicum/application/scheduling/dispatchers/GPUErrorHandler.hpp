#pragma once
#include "services/scheduling/gpu/GPUSink.hpp"

namespace axis { namespace application { namespace scheduling { namespace dispatchers {

/**
 * Implements handling for error events during scheduling or execution in GPU.
 */
class GPUErrorHandler : public axis::services::scheduling::gpu::GPUSink
{
public:
  GPUErrorHandler(void);
  ~GPUErrorHandler(void);

  virtual void Fail( void );
};

} } } } // namespace axis::application::scheduling::dispatchers
