#include "SummaryElementScalarCollector.hpp"
#include <assert.h>
#include <boost/detail/limits.hpp>
#include "application/output/recordsets/ResultRecordset.hpp"
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/collections/ElementSet.hpp"

namespace aaoc = axis::application::output::collectors;
namespace aaocs = axis::application::output::collectors::summarizers;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace adam = axis::domain::algorithms::messages;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;
namespace afb = axis::foundation::blas;

aaocs::SummaryElementScalarCollector::SummaryElementScalarCollector( const axis::String& targetSetName,
                                                            SummaryType summaryType ) :
targetSetName_(targetSetName), summaryType_(summaryType)
{
  // nothing to do here
}

aaocs::SummaryElementScalarCollector::~SummaryElementScalarCollector( void )
{
  // nothing to do here
}

void aaocs::SummaryElementScalarCollector::Collect( 
  const asmm::ResultMessage& message, aaor::ResultRecordset& recordset, 
  const ada::NumericalModel& numericalModel )
{
  adc::ElementSet& set = targetSetName_.empty()? numericalModel.Elements() :
    numericalModel.GetElementSet(targetSetName_);
  size_type count = set.Count();

  StartCollect();
  for (size_type idx = 0; idx < count; ++idx)
  {
    const ade::FiniteElement& element = set.GetByPosition(idx);
    real v = CollectValue(message, element, numericalModel);
    switch (summaryType_)
    {
    case kSum:
    case kAverage:
      value_ += v;
      break;
    case kMaximum:
      if (v > value_) value_ = v;
      break;
    case kMinimum:
      if (v < value_) value_ = v;
      break;
    default:
      assert(!_T("Unexpected summary type!"));
      break;
    }
  }
  if (summaryType_ == kAverage)
  {
    value_ /= (real)count;
  }
  recordset.WriteData(value_);
}

bool aaocs::SummaryElementScalarCollector::IsOfInterest( const asmm::ResultMessage& message ) const
{
  return adam::ModelStateUpdateMessage::IsOfKind(message);
}

void aaocs::SummaryElementScalarCollector::StartCollect( void )
{
  for (int i = 0; i < 6; i++)
  {
    switch (summaryType_)
    {
    case kSum:
    case kAverage:
      value_ = 0;
      break;
    case kMaximum:
      value_ = std::numeric_limits<real>::min();
      break;
    case kMinimum:
      value_ = std::numeric_limits<real>::max();
      break;
    default:
      assert(!_T("Unexpected summary type!"));
      break;
    }
  }
}

axis::String aaocs::SummaryElementScalarCollector::GetFriendlyDescription( void ) const
{
  axis::String desc;
  switch (summaryType_)
  {
  case kSum:
    desc = _T("Total ");
    break;
  case kAverage:
    desc = _T("Average ");
    break;
  case kMaximum:
    desc = _T("Maximum ");
    break;
  case kMinimum:
    desc = _T("Minimum ");
    break;
  default:
    assert(!_T("Unexpected summary type!"));
    break;
  }
  desc += GetVariableName();
  if (targetSetName_.empty())
  {
    desc += _T(" on all elements");
  }
  else
  {
    desc += _T(" on element set '") + targetSetName_ + _T("'");
  }
  return desc;
}
