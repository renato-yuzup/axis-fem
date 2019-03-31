#include "CollectorParseResult.hpp"
#include "foundation/OutOfBoundsException.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aaocs = axis::application::output::collectors::summarizers;
namespace aslp = axis::services::language::parsing;

aafc::CollectorParseResult::CollectorParseResult( const aslp::ParseResult& result ) :
parseResult_(result), collectorType_(kUndefined), groupingType_(aaocs::kNone), 
actOnWholeModel_(true), shouldScale_(false), scaleFactor_((real)1.0)
{
  bool d[6] = {true, true, true, true, true, true};
  Init(d);
}

aafc::CollectorParseResult::CollectorParseResult( const aslp::ParseResult& result, 
                                                  aafc::CollectorType collectorType, 
                                                  aaocs::SummaryType groupingType, 
                                                  const bool *directionState, 
                                                  const axis::String& targetSetName, 
                                                  bool actOnWholeSet, real scaleFactor, 
                                                  bool useScale ) :
parseResult_(result), collectorType_(collectorType), groupingType_(groupingType), 
targetSetName_(targetSetName), actOnWholeModel_(actOnWholeSet), shouldScale_(useScale), 
scaleFactor_(scaleFactor)
{
  Init(directionState);
}

aafc::CollectorParseResult::CollectorParseResult( const CollectorParseResult& other )
{
  operator =(other);
}

aafc::CollectorParseResult& aafc::CollectorParseResult::operator=( const CollectorParseResult& other )
{
  parseResult_ = other.parseResult_;
  collectorType_ = other.collectorType_;
  groupingType_ = other.groupingType_;
  targetSetName_ = other.targetSetName_;
  actOnWholeModel_ = other.actOnWholeModel_;
  shouldScale_ = other.shouldScale_;
  scaleFactor_ = other.scaleFactor_;
  Init(other.directionState_);
  return *this;
}

void aafc::CollectorParseResult::Init( const bool * directionState )
{
  switch (collectorType_)
  {
  case kArtificialEnergy:
  case kEffectivePlasticStrain:
  case kDeformationGradient:
    directionCount_ = 0;
    return;
  case kStress: 
  case kStrain:
  case kPlasticStrain:
    directionCount_ = 6;
    break;
  default:
    directionCount_ = 3;
    break;
  }
  for (int i = 0; i < directionCount_; ++i)
  {
    directionState_[i] = directionState[i];
  }
}

aafc::CollectorParseResult::~CollectorParseResult( void )
{
  // nothing to do here
}

aslp::ParseResult aafc::CollectorParseResult::GetParseResult( void ) const
{
  return parseResult_;
}

aafc::CollectorType aafc::CollectorParseResult::GetCollectorType( void ) const
{
  return collectorType_;
}

aaocs::SummaryType aafc::CollectorParseResult::GetGroupingType( void ) const
{
  return groupingType_;
}

axis::String aafc::CollectorParseResult::GetTargetSetName( void ) const
{
  return targetSetName_;
}

real aafc::CollectorParseResult::GetScaleFactor( void ) const
{
  return scaleFactor_;
}

bool aafc::CollectorParseResult::DoesActOnWholeModel( void ) const
{
  return actOnWholeModel_;
}

bool aafc::CollectorParseResult::ShouldCollectDirection( int directionIndex ) const
{
  switch (collectorType_)
  {
  case kStress:
  case kStrain:
  case kPlasticStrain:
    if (directionIndex < 0 || directionIndex >= 6)
    {
      throw axis::foundation::OutOfBoundsException();
    }
    break;
  case kArtificialEnergy:
  case kEffectivePlasticStrain:
    throw axis::foundation::OutOfBoundsException();
  default:
    if (directionIndex < 0 || directionIndex >= 3)
    {
      throw axis::foundation::OutOfBoundsException();
    }
    break;
  }
  return directionState_[directionIndex];
}

bool aafc::CollectorParseResult::ShouldScaleResults( void ) const
{
  return shouldScale_;
}

int aafc::CollectorParseResult::GetDirectionCount( void ) const
{
  return directionCount_;
}
