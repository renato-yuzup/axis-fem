#if defined(DEBUG) || defined(_DEBUG)
#include "MockResultBucket.hpp"
#include "application/output/ChainMetadata.hpp"

namespace aao = axis::application::output;
namespace adal = axis::domain::analyses;
namespace asmm = axis::services::messaging;
namespace aup = axis::unit_tests::orange;

aup::MockResultBucket::MockResultBucket(void)
{
  // nothing to do here
}

aup::MockResultBucket::~MockResultBucket(void)
{
  // nothing to do here
}

void aup::MockResultBucket::Destroy( void ) const
{
  // nothing to do here
}

void aup::MockResultBucket::PlaceResult( const asmm::ResultMessage& message, 
                                         const adal::NumericalModel& numericalModel )
{
  // nothing to do here
}

aao::ChainMetadata aup::MockResultBucket::GetChainMetadata( int index ) const
{
  return aao::ChainMetadata();
}

int aup::MockResultBucket::GetChainCount( void ) const
{
  return 0;
}
#endif
