#include "SummaryNode6DCollector.hpp"
#include <assert.h>
#include <boost/detail/limits.hpp>
#include "application/output/recordsets/ResultRecordset.hpp"
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/collections/NodeSet.hpp"
#if !defined(AXIS_NO_MEMORY_ARENA)
#include "System.hpp"
#include "foundation/memory/RelativePointer.hpp"
#endif

namespace aaoc = axis::application::output::collectors;
namespace aaocs = axis::application::output::collectors::summarizers;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace adam = axis::domain::algorithms::messages;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;
namespace afb = axis::foundation::blas;
namespace afm = axis::foundation::memory;

aaocs::SummaryNode6DCollector::SummaryNode6DCollector( const axis::String& targetSetName,
                                                       SummaryType summaryType ) :
targetSetName_(targetSetName), summaryType_(summaryType)
{
  for (int i = 0; i < 6; i++) state_[i] = true;
  vectorSize_ = 6;
}
aaocs::SummaryNode6DCollector::SummaryNode6DCollector( const axis::String& targetSetName, 
                                                       SummaryType summaryType,
                                                       aaoc::XXDirectionState xxState,
                                                       aaoc::YYDirectionState yyState,
                                                       aaoc::ZZDirectionState zzState,
                                                       aaoc::YZDirectionState yzState,
                                                       aaoc::XZDirectionState xzState,
                                                       aaoc::XYDirectionState xyState ) :
targetSetName_(targetSetName), summaryType_(summaryType)
{
  state_[0] = (xxState == aaoc::kXXEnabled);
  state_[1] = (yyState == aaoc::kYYEnabled);
  state_[2] = (zzState == aaoc::kZZEnabled);
  state_[3] = (yzState == aaoc::kYZEnabled);
  state_[4] = (xzState == aaoc::kXZEnabled);
  state_[5] = (xyState == aaoc::kXYEnabled);
  int count = 0;
  for (int i = 0; i < 6; i++)  count += state_[i]? 1 : 0;
  vectorSize_ = count;
}

aaocs::SummaryNode6DCollector::~SummaryNode6DCollector( void )
{
  // nothing to do here
}

void aaocs::SummaryNode6DCollector::Collect( const asmm::ResultMessage& message, 
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
    for (int i = 0; i < 6; i++)
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
    for (int i = 0; i < 6; i++)
    {
      if (state_[i]) values_[i] /= (real)count;
    }
  }
  Summarize(recordset);
}

bool aaocs::SummaryNode6DCollector::IsOfInterest( const asmm::ResultMessage& message ) const
{
  return adam::ModelStateUpdateMessage::IsOfKind(message);
}

void aaocs::SummaryNode6DCollector::StartCollect( void )
{
  for (int i = 0; i < 6; i++)
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

void aaocs::SummaryNode6DCollector::Summarize( aaor::ResultRecordset& recordset )
{
  int relativeIdx = 0;
  afb::ColumnVector vector(vectorSize_);
  for (int i = 0; i < 6; i++)
  {
    if (state_[i])
    {
      vector(relativeIdx) = values_[i];
      relativeIdx++;
    }
  }
  recordset.WriteData(vector);
}

axis::String aaocs::SummaryNode6DCollector::GetFriendlyDescription( void ) const
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
  if (vectorSize_ != 6)
  {
    axis::String directionStr;
    for (int i = 0; i < 6; i++)
    {
      if (state_[i])
      {
        if (!directionStr.empty()) directionStr += _T(", ");
        switch (i)
        {
        case 0: directionStr += _T("XX"); break;
        case 1: directionStr += _T("YY"); break;
        case 2: directionStr += _T("ZZ"); break;
        case 3: directionStr += _T("YZ"); break;
        case 4: directionStr += _T("XZ"); break;
        case 5: directionStr += _T("XY"); break;
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
