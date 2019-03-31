#include "NodeExternalLoadCollector.hpp"
#include <assert.h>
#include "domain/analyses/ModelDynamics.hpp"
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

aaoc::NodeExternalLoadCollector::NodeExternalLoadCollector( const axis::String& targetSetName, 
                                                            const axis::String& customFieldName )
: NodeSetCollector(targetSetName), fieldName_(customFieldName)
{
  for (int i = 0; i < 3; ++i) collectState_[i] = true;
  vectorLen_ = 3;
  values_ = new real[3];
}

aaoc::NodeExternalLoadCollector::NodeExternalLoadCollector( const axis::String& targetSetName, 
                                                            const axis::String& customFieldName, 
                                                            XDirectionState xState, 
                                                            YDirectionState yState, 
                                                            ZDirectionState zState )
: NodeSetCollector(targetSetName), fieldName_(customFieldName)
{
  collectState_[0] = (xState == kXEnabled);
  collectState_[1] = (yState == kYEnabled);
  collectState_[2] = (zState == kZEnabled);
  vectorLen_ = 0;
  for (int i = 0; i < 3; ++i)
  {
    if (collectState_[i]) vectorLen_++;
  }
  values_ = new real[vectorLen_];
}

aaoc::NodeExternalLoadCollector& aaoc::NodeExternalLoadCollector::Create( const axis::String& targetSetName )
{
  return Create(targetSetName, kXEnabled, kYEnabled, kZEnabled);
}

aaoc::NodeExternalLoadCollector& aaoc::NodeExternalLoadCollector::Create( const axis::String& targetSetName, 
                                                                          XDirectionState xState, 
                                                                          YDirectionState yState, 
                                                                          ZDirectionState zState )
{
  return Create(targetSetName, _T("Displacement"), xState, yState, zState);
}

aaoc::NodeExternalLoadCollector& aaoc::NodeExternalLoadCollector::Create( const axis::String& targetSetName, 
                                                                          const axis::String& customFieldName )
{
  return Create(targetSetName, customFieldName, kXEnabled, kYEnabled, kZEnabled);
}

aaoc::NodeExternalLoadCollector& aaoc::NodeExternalLoadCollector::Create( const axis::String& targetSetName, 
                                                                          const axis::String& customFieldName, 
                                                                          XDirectionState xState, 
                                                                          YDirectionState yState, 
                                                                          ZDirectionState zState )
{
  return *new aaoc::NodeExternalLoadCollector(targetSetName, customFieldName, xState, yState, zState);
}

aaoc::NodeExternalLoadCollector::~NodeExternalLoadCollector( void )
{
  delete [] values_;
}

void aaoc::NodeExternalLoadCollector::Destroy( void ) const
{
  delete this;
}

void aaoc::NodeExternalLoadCollector::Collect( const ade::Node& node, 
                                               const asmm::ResultMessage& message,                                              
                                               aaor::ResultRecordset& recordset)
{
  const adam::ModelStateUpdateMessage& msuMsg = static_cast<const adam::ModelStateUpdateMessage&>(message);
  const afb::ColumnVector& meshLoads = msuMsg.GetMeshDynamicState().ExternalLoads();
  int relativeIdx = 0;
  for (int i = 0; i < 3; ++i)
  {
    if (collectState_[i]) 
    {
      id_type id = node.GetDoF(i).GetId();
      values_[relativeIdx] = meshLoads(id);
      relativeIdx++;
    }
  }
  recordset.WriteData(afb::ColumnVector(vectorLen_, values_));
}

axis::String aaoc::NodeExternalLoadCollector::GetFieldName( void ) const
{
  return fieldName_;
}

aao::DataType aaoc::NodeExternalLoadCollector::GetFieldType( void ) const
{
  return kVector;
}

int aaoc::NodeExternalLoadCollector::GetVectorFieldLength( void ) const
{
  return vectorLen_;
}

axis::String aaoc::NodeExternalLoadCollector::GetFriendlyDescription( void ) const
{
  axis::String description;
  bool collectAll = true;
  for (int i = 0; i < 3; ++i)
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
        description += _T("X");
        break;
      case 1:
        description += _T("Y");
        break;
      case 2:
        description += _T("Z");
        break;
      default:
        assert(!_T("Unexpected behavior!"));
        break;
      }
    }
  }
  if (collectAll)
  {
    description = _T("External load");
  }
  else
  {
    description += _T(" external load");
  }
  if (GetTargetSetName().empty())
  {
    description += _T(" on all nodes");
  }
  else
  {
    description += _T(" on node set '") + GetTargetSetName() + _T("'");
  }
  return description;
}
