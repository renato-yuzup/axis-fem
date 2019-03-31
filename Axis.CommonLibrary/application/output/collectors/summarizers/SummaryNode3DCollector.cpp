#include "SummaryNode3DCollector.hpp"
#include <assert.h>
#include <boost/detail/limits.hpp>
#include "application/output/recordsets/ResultRecordset.hpp"
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/collections/NodeSet.hpp"

namespace aaoc = axis::application::output::collectors;
namespace aaocs = axis::application::output::collectors::summarizers;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace adam = axis::domain::algorithms::messages;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;
namespace afb = axis::foundation::blas;

aaocs::SummaryNode3DCollector::SummaryNode3DCollector( const axis::String& targetSetName,
                                                       SummaryType summaryType ) :
targetSetName_(targetSetName), summaryType_(summaryType)
{
  for (int i = 0; i < 3; i++) state_[i] = true;
  vectorSize_ = 3;
}
aaocs::SummaryNode3DCollector::SummaryNode3DCollector(const axis::String& targetSetName, 
                                                      SummaryType summaryType,
                                                      aaoc::XDirectionState xState,
                                                      aaoc::YDirectionState yState,
                                                      aaoc::ZDirectionState zState) :
targetSetName_(targetSetName), summaryType_(summaryType)
{
  state_[0] = (xState == aaoc::kXEnabled);
  state_[1] = (yState == aaoc::kYEnabled);
  state_[2] = (zState == aaoc::kZEnabled);
  int count = 0;
  for (int i = 0; i < 3; i++) count += state_[i]? 1 : 0;
  vectorSize_ = count;
}

aaocs::SummaryNode3DCollector::~SummaryNode3DCollector( void )
{
  // nothing to do here
}

void aaocs::SummaryNode3DCollector::Collect( const asmm::ResultMessage& message, 
                                           aaor::ResultRecordset& recordset, 
                                           const ada::NumericalModel& numericalModel )
{
  adc::NodeSet& set = targetSetName_.empty()? numericalModel.Nodes() :
                                              numericalModel.GetNodeSet(targetSetName_);
  size_type count = set.Count();

  StartCollect();
  for (size_type idx = 0; idx < count; ++idx)
  {
    const ade::Node& node = set.GetByPosition(idx);
    for (int i = 0; i < 3; i++)
    {
      if (state_[i])
      {
        real v = CollectValue(message, node, i, numericalModel);
        switch (summaryType_)
        {
        case kAverage:
          values_[i] += v;
          break;
        case kMaximum:
          if (v > values_[i]) values_[i] = v;
          break;
        case kMinimum:
          if (v < values_[i]) values_[i] = v;
          break;
        default:
          assert(!_T("Unexpected summary type!"));
          break;
        }
      }
    }
  }
  if (summaryType_ == kAverage)
  {
    for (int i = 0; i < 3; i++)
    {
      if (state_[i]) values_[i] /= (real)count;
    }
  }
  Summarize(recordset);
}

bool aaocs::SummaryNode3DCollector::IsOfInterest( const asmm::ResultMessage& message ) const
{
  return adam::ModelStateUpdateMessage::IsOfKind(message);
}

void aaocs::SummaryNode3DCollector::StartCollect( void )
{
  for (int i = 0; i < 3; i++)
  {
    switch (summaryType_)
    {
    case kAverage:
      values_[i] = 0;
      break;
    case kMaximum:
      values_[i] = std::numeric_limits<real>::min();
      break;
    case kMinimum:
      values_[i] = std::numeric_limits<real>::max();
      break;
    default:
      assert(!_T("Unexpected summary type!"));
      break;
    }
  }
}

void aaocs::SummaryNode3DCollector::Summarize( aaor::ResultRecordset& recordset )
{
  afb::ColumnVector vector(vectorSize_);
  int relativeIdx = 0;
  for (int i = 0; i < 3; i++)
  {
    if (state_[i])
    {
      vector(relativeIdx) = values_[i];
      relativeIdx++;
    }
  }
  recordset.WriteData(vector);
}

axis::String aaocs::SummaryNode3DCollector::GetFriendlyDescription( void ) const
{
  axis::String desc;
  switch (summaryType_)
  {
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
  if (vectorSize_ != 3)
  {
    axis::String directionStr;
    for (int i = 0; i < 3; i++)
    {
      if (state_[i])
      {
        if (!directionStr.empty()) directionStr += _T(", ");
        switch (i)
        {
        case 0: directionStr += _T("X"); break;
        case 1: directionStr += _T("Y"); break;
        case 2: directionStr += _T("Z"); break;
        }
      }
    }
    desc += directionStr + _T(" ");
  }

  desc += GetVariableName(vectorSize_ != 1);
  if (targetSetName_.empty())
  {
    desc += _T(" on all nodes");
  }
  else
  {
    desc += _T(" on node set '") + targetSetName_ + _T("'");
  }
  return desc;
}
