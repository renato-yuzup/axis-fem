#include "NodeStressCollector.hpp"
#include <assert.h>
#include "domain/analyses/ModelKinematics.hpp"
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"
#include "domain/elements/Node.hpp"
#include "application/output/recordsets/ResultRecordset.hpp"

namespace aao = axis::application::output;
namespace aaoc = axis::application::output::collectors;
namespace asmm = axis::services::messaging;
namespace aaor = axis::application::output::recordsets;
namespace adam = axis::domain::algorithms::messages;
namespace ade = axis::domain::elements;
namespace afb = axis::foundation::blas;

aaoc::NodeStressCollector::NodeStressCollector( const axis::String& targetSetName, 
                                                const axis::String& customFieldName )
: NodeSetCollector(targetSetName), fieldName_(customFieldName)
{
  for (int i = 0; i < 6; ++i) collectState_[i] = true;
  vectorLen_ = 6;
  values_ = new real[6];
}

aaoc::NodeStressCollector::NodeStressCollector( const axis::String& targetSetName, 
                                                const axis::String& customFieldName, 
                                                XXDirectionState xxState, YYDirectionState yyState, 
                                                ZZDirectionState zzState, YZDirectionState yzState, 
                                                XZDirectionState xzState, XYDirectionState xyState)
: NodeSetCollector(targetSetName), fieldName_(customFieldName)
{
  collectState_[0] = (xxState == kXXEnabled);
  collectState_[1] = (yyState == kYYEnabled);
  collectState_[2] = (zzState == kZZEnabled);
  collectState_[3] = (yzState == kYZEnabled);
  collectState_[4] = (xzState == kXZEnabled);
  collectState_[5] = (xyState == kXYEnabled);
  vectorLen_ = 0;
  for (int i = 0; i < 6; ++i)
  {
    if (collectState_[i]) vectorLen_++;
  }
  values_ = new real[vectorLen_];
}

aaoc::NodeStressCollector& aaoc::NodeStressCollector::Create( const axis::String& targetSetName )
{
  return Create(targetSetName, kXXEnabled, kYYEnabled, kZZEnabled, kYZEnabled, kXZEnabled, kXYEnabled);
}

aaoc::NodeStressCollector& aaoc::NodeStressCollector::Create( const axis::String& targetSetName, 
                                                             XXDirectionState xxState, 
                                                             YYDirectionState yyState, 
                                                             ZZDirectionState zzState, 
                                                             YZDirectionState yzState, 
                                                             XZDirectionState xzState,
                                                             XYDirectionState xyState )
{
  return Create(targetSetName, _T("Stress"), xxState, yyState, zzState, yzState, xzState, xyState);
}

aaoc::NodeStressCollector& aaoc::NodeStressCollector::Create( const axis::String& targetSetName, 
                                                              const axis::String& customFieldName )
{
  return Create(targetSetName, customFieldName, 
                kXXEnabled, kYYEnabled, kZZEnabled, kYZEnabled, kXZEnabled, kXYEnabled);
}

aaoc::NodeStressCollector& aaoc::NodeStressCollector::Create( const axis::String& targetSetName, 
                                                              const axis::String& customFieldName, 
                                                              XXDirectionState xxState, 
                                                              YYDirectionState yyState, 
                                                              ZZDirectionState zzState, 
                                                              YZDirectionState yzState, 
                                                              XZDirectionState xzState,
                                                              XYDirectionState xyState )
{
  return *new aaoc::NodeStressCollector(targetSetName, customFieldName, 
                                        xxState, yyState, zzState, yzState, xzState, xyState);
}

aaoc::NodeStressCollector::~NodeStressCollector( void )
{
  delete [] values_;
}

void aaoc::NodeStressCollector::Destroy( void ) const
{
  delete this;
}

void aaoc::NodeStressCollector::Collect( const ade::Node& node, 
                                         const asmm::ResultMessage&,                                              
                                         aaor::ResultRecordset& recordset)
{
  const afb::ColumnVector& nodeStress = node.Stress();
  int relativeIdx = 0;
  for (int i = 0; i < 6; ++i)
  {
    if (collectState_[i]) 
    {
      values_[relativeIdx] = nodeStress(i);
      relativeIdx++;
    }
  }  
  recordset.WriteData(afb::ColumnVector(vectorLen_, values_));
}

axis::String aaoc::NodeStressCollector::GetFieldName( void ) const
{
  return fieldName_;
}

aao::DataType aaoc::NodeStressCollector::GetFieldType( void ) const
{
  return kVector;
}

int aaoc::NodeStressCollector::GetVectorFieldLength( void ) const
{
  return vectorLen_;
}

axis::String aaoc::NodeStressCollector::GetFriendlyDescription( void ) const
{
  axis::String description;
  bool collectAll = true;
  for (int i = 0; i < 6; ++i)
  {
    collectAll = collectAll && collectState_[i];
    if (collectState_[i])
    {
      if (!description.empty())
      {
        description += _T(", ");
      }
      switch (i)
      {
      case 0:
        description += _T("XX");
        break;
      case 1:
        description += _T("YY");
        break;
      case 2:
        description += _T("ZZ");
        break;
      case 3:
        description += _T("YZ");
        break;
      case 4:
        description += _T("XZ");
        break;
      case 5:
        description += _T("XY");
        break;
      default:
        assert(!_T("Unexpected behavior!"));
        break;
      }
    }
  }
  if (collectAll)
  {
    description = _T("Stress tensor");
  }
  else
  {
    description += _T(" stress");
  }
  if (GetTargetSetName().empty())
  {
    description += _T(" of all nodes");
  }
  else
  {
    description += _T(" of node set '") + GetTargetSetName() + _T("'");
  }
  return description;
}
