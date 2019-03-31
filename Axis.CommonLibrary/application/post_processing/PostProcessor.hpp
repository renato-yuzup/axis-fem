#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "services/messaging/ResultMessage.hpp"

namespace axis { namespace application { namespace post_processing {

class AXISCOMMONLIBRARY_API PostProcessor
{
public:
  PostProcessor(void);
  ~PostProcessor(void);

  void ProcessResult(axis::services::messaging::ResultMessage& resultMessage);
};

} } } // namespace axis::application::post_processing
