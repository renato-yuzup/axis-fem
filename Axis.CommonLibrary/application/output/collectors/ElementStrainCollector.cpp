#include "ElementStrainCollector.hpp"
#include <assert.h>
#include "domain/analyses/ModelKinematics.hpp"
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "application/output/recordsets/ResultRecordset.hpp"

namespace aao = axis::application::output;
namespace aaoc = axis::application::output::collectors;
namespace asmm = axis::services::messaging;
namespace aaor = axis::application::output::recordsets;
namespace adam = axis::domain::algorithms::messages;
namespace ade = axis::domain::elements;
namespace afb = axis::foundation::blas;

aaoc::ElementStrainCollector::ElementStrainCollector( const axis::String& targetSetName, 
                                                      const axis::String& customFieldName )
: ElementSetCollector(targetSetName), fieldName_(customFieldName)
{
  for (int i = 0; i < 6; ++i) collectState_[i] = true;
  vectorLen_ = 6;
  values_ = new real[6];
}

aaoc::ElementStrainCollector::ElementStrainCollector( const axis::String& targetSetName, 
                                                      const axis::String& customFieldName, 
                                                      XXDirectionState xxState, YYDirectionState yyState, 
                                                      ZZDirectionState zzState, YZDirectionState yzState, 
                                                      XZDirectionState xzState, XYDirectionState xyState)
: ElementSetCollector(targetSetName), fieldName_(customFieldName)
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

aaoc::ElementStrainCollector& aaoc::ElementStrainCollector::Create( const axis::String& targetSetName )
{
  return Create(targetSetName, kXXEnabled, kYYEnabled, kZZEnabled, kYZEnabled, kXZEnabled, kXYEnabled);
}

aaoc::ElementStrainCollector& aaoc::ElementStrainCollector::Create( const axis::String& targetSetName, 
                                                                    XXDirectionState xxState, 
                                                                    YYDirectionState yyState, 
                                                                    ZZDirectionState zzState, 
                                                                    YZDirectionState yzState, 
                                                                    XZDirectionState xzState, 
                                                                    XYDirectionState xyState)
{
  return Create(targetSetName, _T("Strain"), xxState, yyState, zzState, yzState, xzState, xyState);
}

aaoc::ElementStrainCollector& aaoc::ElementStrainCollector::Create( const axis::String& targetSetName, 
                                                                    const axis::String& customFieldName )
{
  return Create(targetSetName, customFieldName, 
                kXXEnabled, kYYEnabled, kZZEnabled, kYZEnabled, kXZEnabled, kXYEnabled);
}

aaoc::ElementStrainCollector& aaoc::ElementStrainCollector::Create( const axis::String& targetSetName, 
                                                                    const axis::String& customFieldName, 
                                                                    XXDirectionState xxState, 
                                                                    YYDirectionState yyState, 
                                                                    ZZDirectionState zzState, 
                                                                    YZDirectionState yzState, 
                                                                    XZDirectionState xzState, 
                                                                    XYDirectionState xyState)
{
  return *new aaoc::ElementStrainCollector(targetSetName, customFieldName, 
                                           xxState, yyState, zzState, yzState, xzState, xyState);
}

aaoc::ElementStrainCollector::~ElementStrainCollector( void )
{
  delete [] values_;
}

void aaoc::ElementStrainCollector::Destroy( void ) const
{
  delete this;
}

void aaoc::ElementStrainCollector::Collect( const ade::FiniteElement& element, 
                                            const asmm::ResultMessage&,                                              
                                            aaor::ResultRecordset& recordset)
{
  const afb::ColumnVector& nodeStress = element.PhysicalState().Strain();
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

axis::String aaoc::ElementStrainCollector::GetFieldName( void ) const
{
  return fieldName_;
}

aao::DataType aaoc::ElementStrainCollector::GetFieldType( void ) const
{
  return kVector;
}

int aaoc::ElementStrainCollector::GetVectorFieldLength( void ) const
{
  return vectorLen_;
}

axis::String aaoc::ElementStrainCollector::GetFriendlyDescription( void ) const
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
    description = _T("Strain tensor");
  }
  else
  {
    description += _T(" strain");
  }
  if (GetTargetSetName().empty())
  {
    description += _T(" of all elements");
  }
  else
  {
    description += _T(" of element set '") + GetTargetSetName() + _T("'");
  }
  return description;
}
