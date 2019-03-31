#include "ParserAgentImpl.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "foundation/date_time/Timestamp.hpp"

namespace aaa = axis::application::agents;
namespace aaj = axis::application::jobs;
namespace ada = axis::domain::analyses;
namespace afdt = axis::foundation::date_time;
namespace asmg = axis::services::management;

aaa::ParserAgent::ParserAgentImpl::ParserAgentImpl( asmg::GlobalProviderCatalog& catalog ) :
	catalog_(&catalog)
{
	engine_.SetUp(catalog);
}

aaa::ParserAgent::ParserAgentImpl::~ParserAgentImpl( void )
{
	// nothing to do here
}

void aaa::ParserAgent::ParserAgentImpl::AddPreProcessorSymbol( const axis::String& symbolName )
{
	engine_.AddPreProcessorSymbol(symbolName);
  if (!symbolList_.empty())
  {
    symbolList_.append(_T(", "));
  }
  symbolList_.append(symbolName);
}

void aaa::ParserAgent::ParserAgentImpl::ClearPreProcessorSymbols( void )
{
	engine_.ClearPreProcessorSymbols();
  symbolList_.clear();
}

void aaa::ParserAgent::ParserAgentImpl::BuildAnalysis( const axis::String& outputPath )
{
  ada::NumericalModel& model = ada::NumericalModel::Create();
  aaj::StructuralAnalysis& analysis = *new aaj::StructuralAnalysis(outputPath);
  analysis.SetNumericalModel(model);
  analysis.SetCreationDate(afdt::Timestamp::GetUTCTime());
  analysis_ = &analysis;
}

aaj::StructuralAnalysis& aaa::ParserAgent::ParserAgentImpl::GetAnalysis( void ) const
{
	if (analysis_ == NULL)
	{
		throw axis::foundation::InvalidOperationException();
	}
	return *analysis_;
}

void aaa::ParserAgent::ParserAgentImpl::Parse( const axis::String& masterFileName, 
                                               const axis::String& baseIncludePath )
{
	engine_.ConnectListener(*this);
	engine_.Parse(*analysis_, masterFileName, baseIncludePath);
  engine_.DisconnectListener(*this);
}

axis::String aaa::ParserAgent::ParserAgentImpl::GetPreProcessorSymbolList( void ) const
{
  return symbolList_;
}
