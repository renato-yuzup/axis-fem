#include "SummaryElementMatrixCollector.hpp"
#include <boost/detail/limits.hpp>
#include "application/output/recordsets/ResultRecordset.hpp"
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/collections/ElementSet.hpp"

namespace aaocs = axis::application::output::collectors::summarizers;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;
namespace adam = axis::domain::algorithms::messages;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;
namespace afb = axis::foundation::blas;

aaocs::SummaryElementMatrixCollector::SummaryElementMatrixCollector( 
  const axis::String& targetSetName, SummaryType summaryType, int rowCount, 
  int colCount ) : targetSetName_(targetSetName), summaryType_(summaryType)
{
  rowCount_ = rowCount; colCount_ = colCount;
}

aaocs::SummaryElementMatrixCollector::~SummaryElementMatrixCollector( void )
{
  // nothing to do here
}

bool aaocs::SummaryElementMatrixCollector::IsOfInterest( 
  const asmm::ResultMessage& message ) const
{
  return adam::ModelStateUpdateMessage::IsOfKind(message);
}

void aaocs::SummaryElementMatrixCollector::Collect( 
  const asmm::ResultMessage& message, aaor::ResultRecordset& recordset, 
  const ada::NumericalModel& numericalModel )
{
  adc::ElementSet& set = targetSetName_.empty()? numericalModel.Elements() : 
    numericalModel.GetElementSet(targetSetName_);
  size_type count = set.Count();
  afb::DenseMatrix m(rowCount_, colCount_);

  StartCollect(m);
  for (size_type i = 0; i < count; i++)
  {
    const ade::FiniteElement& e = set.GetByPosition(i);
    switch (summaryType_)
    {
    case kAverage:
    case kSum:
      {
        auto& r = CollectMatrix(message, e);
        m += r;
      }
      break;
    case kMaximum:
      {
        real norm = CalculateMatrixNorm(message, e);
        if (norm > bestNormValue_)
        {
          targetIdToCollect_ = i;
          bestNormValue_ = norm;
        }
      }
      break;
    case kMinimum:
      {
        real norm = CalculateMatrixNorm(message, e);
        if (norm < bestNormValue_)
        {
          targetIdToCollect_ = i;
          bestNormValue_ = norm;
        }
      }
      break;
    }
  }

  switch (summaryType_)
  {
  case kAverage:
    m.Scale(1.0 / (real) count);
    break;
  case kMaximum:
  case kMinimum:
    ade::FiniteElement& e = set.GetByPosition(targetIdToCollect_);
    auto& r = CollectMatrix(message, e);
    m = r;
    break;
  }

  recordset.WriteData(m);
}

axis::String aaocs::SummaryElementMatrixCollector::GetFriendlyDescription( 
  void ) const
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

void aaocs::SummaryElementMatrixCollector::StartCollect( 
  afb::DenseMatrix& matrix )
{
  targetIdToCollect_ = -1;
  switch (summaryType_)
  {
  case kSum:
  case kAverage:
    matrix.ClearAll();
    break;
  case kMaximum:
    bestNormValue_ = std::numeric_limits<real>::min();
    break;
  case kMinimum:
    bestNormValue_ = std::numeric_limits<real>::max();
    break;
  }
}
