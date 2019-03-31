#include "CollectorBuildResult.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aaoc = axis::application::output::collectors;
namespace aslp = axis::services::language::parsing;

aafc::CollectorBuildResult::CollectorBuildResult( aaoc::EntityCollector *collector, 
                                                  EntityType collectorType,
                                                  const aslp::ParseResult& result) : 
Collector(collector), CollectorType(collectorType), Result(result), IsModelIncomplete(false)
{
  // nothing to do here
}

aafc::CollectorBuildResult::CollectorBuildResult( aaoc::EntityCollector *collector, 
                                                  EntityType collectorType, 
                                                  const aslp::ParseResult& result, 
                                                  bool modelIncomplete ) :
Collector(collector), CollectorType(collectorType), Result(result), IsModelIncomplete(modelIncomplete)
{
  // nothing to do here
}

aafc::CollectorBuildResult::~CollectorBuildResult( void )
{
  // nothing to do here
}
