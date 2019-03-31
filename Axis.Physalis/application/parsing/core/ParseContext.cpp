#include "ParseContext.hpp"
#include "ParseContext_Pimpl.hpp"
#include "EventStatistic_Pimpl.hpp"
#include "SymbolTable.hpp"
#include "SymbolTable_Pimpl.hpp"
#include "EntityLabeler.hpp"
#include "services/messaging/Metadata/SourceFileMetadata.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aapc = axis::application::parsing::core;
namespace asmm = axis::services::messaging;
namespace asmmm = axis::services::messaging::metadata;

const int aapc::ParseContext::MaxAllowableErrorCount = 5000;
const int aapc::ParseContext::SourceId = 10000;

aapc::ParseContext::ParseContext( void )
{
	pimpl_ = new Pimpl();
  pimpl_->symbols = new SymbolTable(*this);
  pimpl_->sketchbook = new Sketchbook();
}

aapc::ParseContext::~ParseContext( void )
{
  delete pimpl_->symbols;
  delete pimpl_->sketchbook;
	delete pimpl_;
  pimpl_ = NULL;
}

void aapc::ParseContext::RegisterEvent( asmm::EventMessage& event ) const
{
  if (event.IsError())
  {
    pimpl_->eventStatistics.pimpl_->errorEventCount++;
  }
  else if (event.IsWarning())
  {
    pimpl_->eventStatistics.pimpl_->warningEventCount++;
  }
  else if (event.IsInfo())
  {
    pimpl_->eventStatistics.pimpl_->infoEventCount++;
  }

  // add source information to message, if possible
  if (!event.GetMetadata().Contains(asmmm::SourceFileMetadata::GetClassName()))
  {
    event.GetMetadata().Add(asmmm::SourceFileMetadata(GetParseSourceName(), 
                                                      GetParseSourceCursorLocation()));
  }

  if (pimpl_->eventStatistics.pimpl_->lastEvent != NULL)
  {
    pimpl_->eventStatistics.pimpl_->lastEvent->Destroy();
  }
  pimpl_->eventStatistics.pimpl_->lastEvent = (asmm::EventMessage *)&event.Clone();
  // forward message
  DispatchMessage(event);
}

const aapc::ParseContext::EventStatistic& aapc::ParseContext::EventSummary( void ) const
{
	return pimpl_->eventStatistics;
}

aapc::SymbolTable& aapc::ParseContext::Symbols( void )
{
  return *pimpl_->symbols;
}

aapc::EntityLabeler& aapc::ParseContext::Labels( void )
{
  return pimpl_->labeler;
}

const aapc::SymbolTable& aapc::ParseContext::Symbols( void ) const
{
  return *pimpl_->symbols;
}

const aapc::EntityLabeler& aapc::ParseContext::Labels( void ) const
{
  return pimpl_->labeler;
}

void aapc::ParseContext::ClearEventStatistics( void )
{
  pimpl_->eventStatistics.pimpl_->errorEventCount = 0;
  pimpl_->eventStatistics.pimpl_->warningEventCount = 0;
  pimpl_->eventStatistics.pimpl_->infoEventCount = 0;
  if (pimpl_->eventStatistics.pimpl_->lastEvent != NULL)
  {
    pimpl_->eventStatistics.pimpl_->lastEvent->Destroy();
    pimpl_->eventStatistics.pimpl_->lastEvent = NULL;
  }
}

void aapc::ParseContext::AdvanceRound( void )
{
  pimpl_->symbols->pimpl_->autoNames.clear();
  pimpl_->symbols->pimpl_->definedRefCount = 0;
  pimpl_->symbols->pimpl_->unresolvedRefCount = 0;
}

aapc::Sketchbook& aapc::ParseContext::Sketches( void )
{
  return *pimpl_->sketchbook;
}

const aapc::Sketchbook& aapc::ParseContext::Sketches( void ) const
{
  return *pimpl_->sketchbook;
}
