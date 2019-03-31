#include "NodeReactionForceCollector.hpp"
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

aaoc::NodeReactionForceCollector::NodeReactionForceCollector( const axis::String& targetSetName, 
                                                              const axis::String& customFieldName )
: NodeSetCollector(targetSetName), fieldName_(customFieldName)
{
  for (int i = 0; i < 3; ++i) collectState_[i] = true;
  vectorLen_ = 3;
  values_ = new real[3];
}

aaoc::NodeReactionForceCollector::NodeReactionForceCollector( const axis::String& targetSetName, 
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

aaoc::NodeReactionForceCollector& aaoc::NodeReactionForceCollector::Create( const axis::String& targetSetName )
{
  return Create(targetSetName, kXEnabled, kYEnabled, kZEnabled);
}

aaoc::NodeReactionForceCollector& aaoc::NodeReactionForceCollector::Create( const axis::String& targetSetName, 
                                                                            XDirectionState xState, 
                                                                            YDirectionState yState, 
                                                                            ZDirectionState zState )
{
  return Create(targetSetName, _T("ReactionForce"), xState, yState, zState);
}

aaoc::NodeReactionForceCollector& aaoc::NodeReactionForceCollector::Create( const axis::String& targetSetName, 
                                                                            const axis::String& customFieldName )
{
  return Create(targetSetName, customFieldName, kXEnabled, kYEnabled, kZEnabled);
}

aaoc::NodeReactionForceCollector& aaoc::NodeReactionForceCollector::Create( const axis::String& targetSetName, 
                                                                            const axis::String& customFieldName, 
                                                                            XDirectionState xState, 
                                                                            YDirectionState yState, 
                                                                            ZDirectionState zState )
{
  return *new aaoc::NodeReactionForceCollector(targetSetName, customFieldName, xState, yState, zState);
}

aaoc::NodeReactionForceCollector::~NodeReactionForceCollector( void )
{
  delete [] values_;
}

void aaoc::NodeReactionForceCollector::Destroy( void ) const
{
  delete this;
}

void aaoc::NodeReactionForceCollector::Collect( const ade::Node& node, 
                                                const asmm::ResultMessage& message,                                              
                                                aaor::ResultRecordset& recordset)
{
  const adam::ModelStateUpdateMessage& msuMsg = static_cast<const adam::ModelStateUpdateMessage&>(message);
  const afb::ColumnVector& meshReaction = msuMsg.GetMeshDynamicState().ReactionForce();
  int relativeIdx = 0;
  for (int i = 0; i < 3; ++i)
  {
    if (collectState_[i]) 
    {
      id_type id = node.GetDoF(i).GetId();
      values_[relativeIdx] = meshReaction(id);
      relativeIdx++;
    }
  }  
  recordset.WriteData(afb::ColumnVector(vectorLen_, values_));
}

axis::String aaoc::NodeReactionForceCollector::GetFieldName( void ) const
{
  return fieldName_;
}

aao::DataType aaoc::NodeReactionForceCollector::GetFieldType( void ) const
{
  return kVector;
}

int aaoc::NodeReactionForceCollector::GetVectorFieldLength( void ) const
{
  return vectorLen_;
}

axis::String aaoc::NodeReactionForceCollector::GetFriendlyDescription( void ) const
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
    description = _T("Reaction force");
  }
  else
  {
    description += _T(" reaction force");
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
