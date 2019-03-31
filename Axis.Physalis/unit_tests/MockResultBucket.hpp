#pragma once
#include "application/output/ResultBucket.hpp"

namespace axis { namespace unit_tests { namespace physalis {

class MockResultBucket : public axis::application::output::ResultBucket
{
public:
  MockResultBucket(void);
  ~MockResultBucket(void);
  virtual void Destroy( void ) const;
  virtual void PlaceResult( const axis::services::messaging::ResultMessage& message, 
                            const axis::domain::analyses::NumericalModel& numericalModel );
  virtual axis::application::output::ChainMetadata GetChainMetadata( int index ) const;
  virtual int GetChainCount( void ) const;
};

} } } // namespace axis::unit_tests::physalis