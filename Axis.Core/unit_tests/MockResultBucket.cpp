#if defined(DEBUG) || defined(_DEBUG)
#include "MockResultBucket.hpp"
#include "application/output/ChainMetadata.hpp"

namespace aao = axis::application::output;
namespace adal = axis::domain::analyses;
namespace asmm = axis::services::messaging;

MockResultBucket::MockResultBucket(void)
{
  // nothing to do here
}

MockResultBucket::~MockResultBucket(void)
{
  // nothing to do here
}

void MockResultBucket::Destroy( void ) const
{
  // nothing to do here
}

void MockResultBucket::PlaceResult( const asmm::ResultMessage& message, 
                                    const adal::NumericalModel& numericalModel )
{
  // nothing to do here
}

aao::ChainMetadata MockResultBucket::GetChainMetadata( int index ) const
{
  return aao::ChainMetadata();
}

int MockResultBucket::GetChainCount( void ) const
{
  return 0;
}
#endif
